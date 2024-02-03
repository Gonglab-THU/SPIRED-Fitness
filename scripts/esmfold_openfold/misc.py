# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as T
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

def batch_encode_sequences(
    sequences: T.Sequence[str],
    residue_index_offset: T.Optional[int] = 512,
    chain_linker: T.Optional[str] = "G" * 25,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    aatype_list = []
    residx_list = []
    linker_mask_list = []
    chain_index_list = []
    for seq in sequences:
        aatype_seq, residx_seq, linker_mask_seq, chain_index_seq = encode_sequence(
            seq,
            residue_index_offset=residue_index_offset,
            chain_linker=chain_linker,
        )
        aatype_list.append(aatype_seq)
        residx_list.append(residx_seq)
        linker_mask_list.append(linker_mask_seq)
        chain_index_list.append(chain_index_seq)

    aatype = collate_dense_tensors(aatype_list)
    mask = collate_dense_tensors(
        [aatype.new_ones(len(aatype_seq)) for aatype_seq in aatype_list]
    )
    residx = collate_dense_tensors(residx_list)
    linker_mask = collate_dense_tensors(linker_mask_list)
    chain_index_list = collate_dense_tensors(chain_index_list, -1)

    return aatype, mask, residx, linker_mask, chain_index_list

def collate_dense_tensors(
    samples: T.List[torch.Tensor], pad_v: float = 0
) -> torch.Tensor:
    """
    Takes a list of tensors with the following dimensions:
        [(d_11,       ...,           d_1K),
         (d_21,       ...,           d_2K),
         ...,
         (d_N1,       ...,           d_NK)]
    and stack + pads them into a single tensor of:
    (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
    """
    if len(samples) == 0:
        return torch.Tensor()
    if len(set(x.dim() for x in samples)) != 1:
        raise RuntimeError(
            f"Samples has varying dimensions: {[x.dim() for x in samples]}"
        )
    (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = torch.empty(
        len(samples), *max_shape, dtype=samples[0].dtype, device=device
    )
    result.fill_(pad_v)
    for i in range(len(samples)):
        result_i = result[i]
        t = samples[i]
        result_i[tuple(slice(0, k) for k in t.shape)] = t
    return result


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_width, gated=False):
        super().__init__()
        assert embed_dim == num_heads * head_width

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_width = head_width

        self.proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gated = gated
        if gated:
            self.g_proj = nn.Linear(embed_dim, embed_dim)
            torch.nn.init.zeros_(self.g_proj.weight)
            torch.nn.init.ones_(self.g_proj.bias)

        self.rescale_factor = self.head_width**-0.5

        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, x, mask=None, bias=None, indices=None):
        """
        Basic self attention with optional mask and external pairwise bias.
        To handle sequences of different lengths, use mask.

        Inputs:
          x: batch of input sequneces (.. x L x C)
          mask: batch of boolean masks where 1=valid, 0=padding position (.. x L_k). optional.
          bias: batch of scalar pairwise attention biases (.. x Lq x Lk x num_heads). optional.

        Outputs:
          sequence projection (B x L x embed_dim), attention maps (B x L x L x num_heads)
        """

        t = rearrange(self.proj(x), "... l (h c) -> ... h l c", h=self.num_heads)
        q, k, v = t.chunk(3, dim=-1)

        q = self.rescale_factor * q
        a = torch.einsum("...qc,...kc->...qk", q, k)

        # Add external attention bias.
        if bias is not None:
            a = a + rearrange(bias, "... lq lk h -> ... h lq lk")

        # Do not attend to padding tokens.
        if mask is not None:
            mask = repeat(
                mask, "... lk -> ... h lq lk", h=self.num_heads, lq=q.shape[-2]
            )
            a = a.masked_fill(mask == False, -np.inf)

        a = F.softmax(a, dim=-1)

        y = torch.einsum("...hqk,...hkc->...qhc", a, v)
        y = rearrange(y, "... h c -> ... (h c)", h=self.num_heads)

        if self.gated:
            y = self.g_proj(x).sigmoid() * y
        y = self.o_proj(y)

        return y, rearrange(a, "... lq lk h -> ... h lq lk")


class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.
    """

    def __init__(self, r: float, batch_dim: T.Union[int, T.List[int]]):
        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        return x * self.dropout(x.new_ones(shape))


class SequenceToPair(nn.Module):
    def __init__(self, sequence_state_dim, inner_dim, pairwise_state_dim):
        super().__init__()

        self.layernorm = nn.LayerNorm(sequence_state_dim)
        self.proj = nn.Linear(sequence_state_dim, inner_dim * 2, bias=True)
        self.o_proj = nn.Linear(2 * inner_dim, pairwise_state_dim, bias=True)

        torch.nn.init.zeros_(self.proj.bias)
        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, sequence_state):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim

        Output:
          pairwise_state: B x L x L x pairwise_state_dim

        Intermediate state:
          B x L x L x 2*inner_dim
        """

        assert len(sequence_state.shape) == 3

        s = self.layernorm(sequence_state)
        s = self.proj(s)
        q, k = s.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]

        x = torch.cat([prod, diff], dim=-1)
        x = self.o_proj(x)

        return x


class PairToSequence(nn.Module):
    def __init__(self, pairwise_state_dim, num_heads):
        super().__init__()

        self.layernorm = nn.LayerNorm(pairwise_state_dim)
        self.linear = nn.Linear(pairwise_state_dim, num_heads, bias=False)

    def forward(self, pairwise_state):
        """
        Inputs:
          pairwise_state: B x L x L x pairwise_state_dim

        Output:
          pairwise_bias: B x L x L x num_heads
        """
        assert len(pairwise_state.shape) == 4
        z = self.layernorm(pairwise_state)
        pairwise_bias = self.linear(z)
        return pairwise_bias


class ResidueMLP(nn.Module):
    def __init__(self, embed_dim, inner_dim, norm=nn.LayerNorm, dropout=0):
        super().__init__()

        self.mlp = nn.Sequential(
            norm(embed_dim),
            nn.Linear(embed_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.mlp(x)
