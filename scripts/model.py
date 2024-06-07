import torch
from torch import nn
from . import Module
from contextlib import ExitStack
from .esmfold_openfold.tri_self_attn_block import TriangularSelfAttentionBlock
from .utils_train_valid import return_predcadist_plddt


class RelativePosition(nn.Module):
    def __init__(self, bins, pairwise_state_dim):
        super().__init__()
        self.bins = bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = torch.nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index, mask=None):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long)
          mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """

        assert residue_index.dtype == torch.long
        if mask is not None:
            assert residue_index.shape == mask.shape

        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.

        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0

        output = self.embedding(diff)
        return output


eps = 1e-8


def getCadistAvg(relativepredxyz):
    # relativepredxyz =(N,3,L,L,L)
    relativepredcadist2 = torch.sum(relativepredxyz * relativepredxyz, dim=1)  # (N,L,L,L), distance square
    relativepredcadist = torch.sqrt(relativepredcadist2 + eps)  # (N,L,L,L)
    cadistavgNLL = torch.mean(relativepredcadist, dim=1)
    return cadistavgNLL


class featureReduction(nn.Module):
    def __init__(self, cin, channel):
        super().__init__()

        self.ircn = Module.IRCN(cin)
        self.conv = nn.Sequential(nn.Conv2d(3 * cin, channel, kernel_size=(1, 1)), nn.ELU())
        self.insNorm = nn.InstanceNorm2d(channel, affine=True)

    def forward(self, feature):
        # feature = (B, L, L, C)
        feature = feature.permute(0, 3, 1, 2).contiguous()
        out = self.ircn(feature)
        out = self.conv(out)
        out_insnorm = self.insNorm(out)
        out = out.permute(0, 2, 3, 1).contiguous()
        out_insnorm = out_insnorm.permute(0, 2, 3, 1).contiguous()
        return [out, out_insnorm]


class BlockV0(nn.Module):
    def __init__(self, depth, channel):
        super().__init__()

        self.pairwise_positional_embedding = RelativePosition(32, 128)

        block = TriangularSelfAttentionBlock

        self.blocks = nn.ModuleList(
            [
                block(
                    sequence_state_dim=1024,
                    pairwise_state_dim=128,
                    sequence_head_width=32,
                    pairwise_head_width=32,
                    dropout=0,
                )
                for i in range(depth)
            ]
        )

        self.ircn0 = featureReduction(channel, channel)
        self.ircn1 = featureReduction(2 * channel, channel)
        self.ircn2 = featureReduction(2 * channel, channel)
        self.plddt0 = Module.pLDDT(channel)

        self.predDxyz0 = Module.predDxyz_shareWeight(channel)
        self.plddt1 = Module.pLDDT(channel)

        self.forNextBlock = Module.forNextBlock(channel)

    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, maskdiag):

        device = seq_feats.device
        Predxyz = []
        PredCadistavg = []
        Plddt = []

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)

            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=None)
            return s, z

        ### Triangular Attention
        seq_feats, pair_feats = trunk_iter(seq_feats, pair_feats, residx, mask)
        residual, pair_feats = self.ircn0(pair_feats)

        #### Predict Ca coordinates first time
        predxyz, pair_feats = self.predDxyz0(pair_feats)

        predxyz = predxyz * maskdiag

        pair_feats = torch.cat((residual, pair_feats), dim=-1)
        residual, pair_feats = self.ircn1(pair_feats)

        relativepredxyz = predxyz[:, :, :, None, :] - predxyz[:, :, :, :, None]  # (N,3,L,L,L)
        cadistavg = getCadistAvg(relativepredxyz)

        Predxyz.append(predxyz)
        PredCadistavg.append(cadistavg)
        Plddt.append(self.plddt0(pair_feats, cadistavg))

        #### Predict Ca coordinates second time
        dxyz, pair_feats = self.predDxyz0(pair_feats, cadistavg)
        dxyz = dxyz * maskdiag

        pair_feats = torch.cat((residual, pair_feats), dim=-1)
        residual, pair_feats = self.ircn2(pair_feats)

        predxyz = predxyz + dxyz
        relativepredxyz = predxyz[:, :, :, None, :] - predxyz[:, :, :, :, None]  # (N,3,L,L,L)
        cadistavg = getCadistAvg(relativepredxyz)

        Predxyz.append(predxyz)
        PredCadistavg.append(cadistavg)
        Plddt.append(self.plddt1(pair_feats, cadistavg))

        pair_feats = self.forNextBlock(pair_feats, cadistavg)

        return Predxyz, PredCadistavg, Plddt, seq_feats, pair_feats


class BlockV1(nn.Module):
    def __init__(self, depth, channel):
        super().__init__()

        self.pairwise_positional_embedding = RelativePosition(32, 128)

        block = TriangularSelfAttentionBlock

        self.blocks = nn.ModuleList(
            [
                block(
                    sequence_state_dim=1024,
                    pairwise_state_dim=128,
                    sequence_head_width=32,
                    pairwise_head_width=32,
                    dropout=0,
                )
                for _ in range(depth)
            ]
        )

        self.ircn0 = featureReduction(channel, channel)
        self.ircn1 = featureReduction(2 * channel, channel)
        self.ircn2 = featureReduction(2 * channel, channel)
        # self.predxyz = Module.predxyz(channel)
        self.plddt0 = Module.pLDDT(channel)

        self.predDxyz0 = Module.predDxyz_shareWeight(channel)  # 264=2*channel+8
        self.plddt1 = Module.pLDDT(channel)

        self.forNextBlock = Module.forNextBlock(channel)

    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, maskdiag):

        Predxyz = []
        PredCadistavg = []
        Plddt = []

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)

            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=None)
            return s, z

        seq_feats, pair_feats = trunk_iter(seq_feats, pair_feats, residx, mask)
        residual, pair_feats = self.ircn0(pair_feats)

        # Predict Ca coordinates first time
        predxyz, pair_feats = self.predDxyz0(pair_feats)
        predxyz = predxyz * maskdiag
        pair_feats = torch.cat((residual, pair_feats), dim=-1)
        residual, pair_feats = self.ircn1(pair_feats)

        relativepredxyz = predxyz[:, :, :, None, :] - predxyz[:, :, :, :, None]  # (N,3,L,L,L)
        cadistavg = getCadistAvg(relativepredxyz)

        Predxyz.append(predxyz)
        PredCadistavg.append(cadistavg)
        Plddt.append(self.plddt0(pair_feats, cadistavg))

        # Predict Ca coordinates second time
        dxyz, pair_feats = self.predDxyz0(pair_feats, cadistavg)
        dxyz = dxyz * maskdiag
        pair_feats = torch.cat((residual, pair_feats), dim=-1)
        residual, pair_feats = self.ircn2(pair_feats)

        predxyz = predxyz + dxyz
        relativepredxyz = predxyz[:, :, :, None, :] - predxyz[:, :, :, :, None]  # (N,3,L,L,L)
        cadistavg = getCadistAvg(relativepredxyz)

        Predxyz.append(predxyz)
        PredCadistavg.append(cadistavg)
        Plddt.append(self.plddt1(pair_feats, cadistavg))

        pair_feats = self.forNextBlock(pair_feats, cadistavg)

        return Predxyz, PredCadistavg, Plddt, seq_feats, pair_feats


class BlockV2(nn.Module):
    def __init__(self, depth, channel):
        super().__init__()

        self.pairwise_positional_embedding = RelativePosition(32, 128)

        block = TriangularSelfAttentionBlock

        self.blocks = nn.ModuleList(
            [
                block(
                    sequence_state_dim=1024,
                    pairwise_state_dim=128,
                    sequence_head_width=32,
                    pairwise_head_width=32,
                    dropout=0,
                )
                for i in range(depth)
            ]
        )

        self.ircn0 = featureReduction(channel, channel)
        self.ircn1 = featureReduction(2 * channel, channel)
        self.ircn2 = featureReduction(2 * channel, channel)
        self.ircn3 = featureReduction(2 * channel, channel)
        self.ircn4 = featureReduction(2 * channel, channel)
        self.ircn5 = featureReduction(2 * channel, channel)
        self.ircn6 = featureReduction(2 * channel, channel)

        self.predDxyz0 = Module.predDxyz_shareWeight(channel)
        self.predDxyz1 = Module.predDxyz_shareWeight(channel)

        self.plddt0 = Module.pLDDT(channel)
        self.plddt1 = Module.pLDDT(channel)
        self.plddt2 = Module.pLDDT(channel)
        self.plddt3 = Module.pLDDT(channel)
        self.plddt4 = Module.pLDDT(channel)
        self.plddt5 = Module.pLDDT(channel)

        self.forNextBlock = Module.forNextBlock(channel)

    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, maskdiag):

        device = seq_feats.device
        Predxyz = []
        PredCadistavg = []
        Plddt = []

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)

            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=None)
            return s, z

        ### Triangular Attention
        seq_feats, pair_feats = trunk_iter(seq_feats, pair_feats, residx, mask)
        residual, pair_feats = self.ircn0(pair_feats)

        # Predict Ca coordinates first time
        predxyz, pair_feats = self.predDxyz0(pair_feats)
        predxyz = predxyz * maskdiag
        pair_feats = torch.cat((residual, pair_feats), dim=-1)
        residual, pair_feats = self.ircn1(pair_feats)

        relativepredxyz = predxyz[:, :, :, None, :] - predxyz[:, :, :, :, None]  # (N,3,L,L,L)
        cadistavg = getCadistAvg(relativepredxyz)

        Predxyz.append(predxyz)
        PredCadistavg.append(cadistavg)
        Plddt.append(self.plddt0(pair_feats, cadistavg))

        # Predict Ca coordinates second time
        dxyz, pair_feats = self.predDxyz0(pair_feats, cadistavg)
        dxyz = dxyz * maskdiag
        pair_feats = torch.cat((residual, pair_feats), dim=-1)
        residual, pair_feats = self.ircn2(pair_feats)

        predxyz = predxyz + dxyz
        relativepredxyz = predxyz[:, :, :, None, :] - predxyz[:, :, :, :, None]  # (N,3,L,L,L)
        cadistavg = getCadistAvg(relativepredxyz)

        Predxyz.append(predxyz)
        PredCadistavg.append(cadistavg)
        Plddt.append(self.plddt1(pair_feats, cadistavg))

        # Predict Ca coordinates third time
        dxyz, pair_feats = self.predDxyz0(pair_feats, cadistavg)
        dxyz = dxyz * maskdiag
        pair_feats = torch.cat((residual, pair_feats), dim=-1)
        residual, pair_feats = self.ircn3(pair_feats)

        predxyz = predxyz + dxyz
        relativepredxyz = predxyz[:, :, :, None, :] - predxyz[:, :, :, :, None]  # (N,3,L,L,L)
        cadistavg = getCadistAvg(relativepredxyz)

        Predxyz.append(predxyz)
        PredCadistavg.append(cadistavg)
        Plddt.append(self.plddt2(pair_feats, cadistavg))

        # Predict Ca coordinates fourth time
        dxyz, pair_feats = self.predDxyz0(pair_feats, cadistavg)
        dxyz = dxyz * maskdiag
        pair_feats = torch.cat((residual, pair_feats), dim=-1)
        residual, pair_feats = self.ircn4(pair_feats)

        predxyz = predxyz + dxyz
        relativepredxyz = predxyz[:, :, :, None, :] - predxyz[:, :, :, :, None]  # (N,3,L,L,L)
        cadistavg = getCadistAvg(relativepredxyz)

        Predxyz.append(predxyz)
        PredCadistavg.append(cadistavg)
        Plddt.append(self.plddt3(pair_feats, cadistavg))

        # Predict Ca coordinates fifth time
        dxyz, pair_feats = self.predDxyz1(pair_feats, cadistavg)
        dxyz = dxyz * maskdiag
        pair_feats = torch.cat((residual, pair_feats), dim=-1)
        residual, pair_feats = self.ircn5(pair_feats)

        predxyz = predxyz + dxyz
        relativepredxyz = predxyz[:, :, :, None, :] - predxyz[:, :, :, :, None]  # (N,3,L,L,L)
        cadistavg = getCadistAvg(relativepredxyz)

        Predxyz.append(predxyz)
        PredCadistavg.append(cadistavg)
        Plddt.append(self.plddt4(pair_feats, cadistavg))

        # Predict Ca coordinates sixth time
        dxyz, pair_feats = self.predDxyz1(pair_feats, cadistavg)
        dxyz = dxyz * maskdiag
        pair_feats = torch.cat((residual, pair_feats), dim=-1)
        residual, pair_feats = self.ircn6(pair_feats)

        predxyz = predxyz + dxyz
        relativepredxyz = predxyz[:, :, :, None, :] - predxyz[:, :, :, :, None]  # (N,3,L,L,L)
        cadistavg = getCadistAvg(relativepredxyz)

        Predxyz.append(predxyz)
        PredCadistavg.append(cadistavg)
        Plddt.append(self.plddt5(pair_feats, cadistavg))

        pair_feats = self.forNextBlock(pair_feats, cadistavg)

        return Predxyz, PredCadistavg, Plddt, seq_feats, pair_feats


class SPIRED_Model(nn.Module):
    def __init__(self, depth, channel, device_list):
        super().__init__()

        device_num = len(device_list)

        if device_num == 4:
            self.device0, self.device1, self.device2, self.device3 = (torch.device(device_list[0]), torch.device(device_list[1]), torch.device(device_list[2]), torch.device(device_list[3]))
        elif device_num == 3:
            self.device0, self.device1, self.device2, self.device3 = (torch.device(device_list[0]), torch.device(device_list[1]), torch.device(device_list[1]), torch.device(device_list[2]))
        elif device_num == 2:
            self.device0, self.device1, self.device2, self.device3 = (torch.device(device_list[0]), torch.device(device_list[0]), torch.device(device_list[1]), torch.device(device_list[1]))
        elif device_num == 1:
            self.device0, self.device1, self.device2, self.device3 = (torch.device(device_list[0]), torch.device(device_list[0]), torch.device(device_list[0]), torch.device(device_list[0]))

        self.esm_s_combine = nn.ParameterDict({"weight": nn.Parameter(torch.zeros(37), requires_grad=True)})
        self.esm_s_combine = self.esm_s_combine.to(self.device0)

        self.esm_s_mlp = nn.Sequential(
            nn.LayerNorm(2560),
            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        ).to(self.device0)

        self.embedding = nn.Embedding(33, 1024, padding_idx=0).to(self.device0)

        self.recycle_bins = 47
        self.recycle_s_norm = nn.LayerNorm(1024).to(self.device0)
        self.recycle_z_norm = nn.LayerNorm(128).to(self.device0)
        self.recycle_disto = nn.Linear(self.recycle_bins, 128).to(self.device0)
        self.recycle_disto.weight[0].detach().zero_()

        ######
        self.block0 = BlockV0(depth, channel).to(self.device0)
        self.block1 = BlockV1(depth, channel).to(self.device1)
        self.block2 = BlockV1(depth, channel).to(self.device2)
        self.block3 = BlockV2(depth, channel).to(self.device3)

        self.featureReduction0 = featureReduction(2 * channel, channel).to(self.device1)
        self.featureReduction1 = featureReduction(2 * channel, channel).to(self.device2)
        self.featureReduction2 = featureReduction(2 * channel, channel).to(self.device3)

        self.toCACBangle = Module.to_CA_CB_Angle(channel).to(self.device3)
        self.to_phi_psi = Module.to_phi_psi(1024).to(self.device3)

    def forward(self, true_aa, f1d, no_recycles, labelbatch=None):
        ### f1d:(N,L,37,2560)

        Predxyz = {}
        PredCadistavg = {}
        Plddt = {}

        if f1d.device == torch.device("cpu"):
            self.device0, self.device1, self.device2, self.device3 = (torch.device("cpu"), torch.device("cpu"), torch.device("cpu"), torch.device("cpu"))

        B, L, layers, c_s = f1d.shape
        maskdiag = torch.ones([L, L]) - torch.eye(L)
        f1d, maskdiag, true_aa = (f1d.to(self.device0), maskdiag.to(self.device0), true_aa.to(self.device0))

        mask = torch.ones_like(true_aa).to(self.device0)
        residx = torch.arange(L, device=self.device0).expand_as(true_aa)
        f1d = (self.esm_s_combine["weight"].softmax(0).unsqueeze(0) @ f1d).squeeze(2)

        s_s_0 = self.esm_s_mlp(f1d)
        s_s_0 += self.embedding(true_aa)

        s_z_0 = s_s_0.new_zeros(B, L, L, 128)

        s_s = s_s_0
        s_z = s_z_0
        recycle_s = torch.zeros_like(s_s).to(self.device0)
        recycle_z = torch.zeros_like(s_z).to(self.device0)
        recycle_bins = torch.zeros((B, L, L, 47), device=self.device0, dtype=f1d.dtype)

        for recycle_idx in range(no_recycles):
            with ExitStack() if recycle_idx == (no_recycles - 1) else torch.no_grad():
                # === Recycling ===
                recycle_s = self.recycle_s_norm(recycle_s.detach().to(self.device0))
                recycle_z = self.recycle_z_norm(recycle_z.detach().to(self.device0))
                recycle_z += self.recycle_disto(recycle_bins.detach().to(self.device0))

                seq_feats = s_s_0 + recycle_s
                pair_feats = s_z_0 + recycle_z
                seq_feats, pair_feats, featureResidual, maskdiag, residx, mask, true_aa = (seq_feats.to(self.device0), pair_feats.to(self.device0), pair_feats.to(self.device0), maskdiag.to(self.device0), residx.to(self.device0), mask.to(self.device0), true_aa.to(self.device0))

                ### Folding Unit1
                Predxyz0, PredCadistavg0, Plddt0, seq_feats, pair_feats = self.block0(seq_feats, pair_feats, true_aa, residx, mask, maskdiag)
                Predxyz["1st"], PredCadistavg["1st"], Plddt["1st"] = Predxyz0, PredCadistavg0, Plddt0
                seq_feats, pair_feats, featureResidual, maskdiag, residx, mask, true_aa = (seq_feats.to(self.device1), pair_feats.to(self.device1), pair_feats.to(self.device1), maskdiag.to(self.device1), residx.to(self.device1), mask.to(self.device1), true_aa.to(self.device1))
                pair_feats = (pair_feats + pair_feats.permute(0, 2, 1, 3)) / 2.0

                ### Folding Unit2
                self.block1, self.featureReduction0 = (self.block1.to(self.device1), self.featureReduction0.to(self.device1))
                Predxyz0, PredCadistavg0, Plddt0, seq_feats, pair_feats = self.block1(seq_feats, pair_feats, true_aa, residx, mask, maskdiag)
                Predxyz["2nd"], PredCadistavg["2nd"], Plddt["2nd"] = Predxyz0, PredCadistavg0, Plddt0
                pair_feats = (pair_feats + pair_feats.permute(0, 2, 1, 3)) / 2.0
                feature = torch.cat((featureResidual, pair_feats), dim=-1)

                featureResidual, pair_feats = self.featureReduction0(feature)

                seq_feats, pair_feats, featureResidual, maskdiag, residx, mask, true_aa = (seq_feats.to(self.device2), pair_feats.to(self.device2), featureResidual.to(self.device2), maskdiag.to(self.device2), residx.to(self.device2), mask.to(self.device2), true_aa.to(self.device2))

                ### Folding Unit3
                self.block2, self.featureReduction1 = (self.block2.to(self.device2), self.featureReduction1.to(self.device2))
                Predxyz0, PredCadistavg0, Plddt0, seq_feats, pair_feats = self.block2(seq_feats, pair_feats, true_aa, residx, mask, maskdiag)
                Predxyz["3rd"], PredCadistavg["3rd"], Plddt["3rd"] = Predxyz0, PredCadistavg0, Plddt0
                pair_feats = (pair_feats + pair_feats.permute(0, 2, 1, 3)) / 2.0
                feature = torch.cat((featureResidual, pair_feats), dim=-1)
                featureResidual, pair_feats = self.featureReduction1(feature)
                seq_feats, pair_feats, featureResidual, maskdiag, residx, mask, true_aa = (seq_feats.to(self.device3), pair_feats.to(self.device3), featureResidual.to(self.device3), maskdiag.to(self.device3), residx.to(self.device3), mask.to(self.device3), true_aa.to(self.device3))

                ### Folding Unit4
                self.block3, self.featureReduction2 = (self.block3.to(self.device3), self.featureReduction2.to(self.device3))
                Predxyz0, PredCadistavg0, Plddt0, seq_feats, pair_feats = self.block3(seq_feats, pair_feats, true_aa, residx, mask, maskdiag)
                Predxyz["4th"], PredCadistavg["4th"], Plddt["4th"] = Predxyz0, PredCadistavg0, Plddt0
                pair_feats = (pair_feats + pair_feats.permute(0, 2, 1, 3)) / 2.0
                feature = torch.cat((featureResidual, pair_feats), dim=-1)
                featureResidual, pair_feats = self.featureReduction2(feature)

                self.toCACBangle, self.to_phi_psi = (self.toCACBangle.to(self.device3), self.to_phi_psi.to(self.device3))
                ca, cb, omega, theta, phi = self.toCACBangle(pair_feats)
                phi_psi_1D = self.to_phi_psi(seq_feats)  # (batch, 2, L-1)

                recycle_s = seq_feats
                recycle_z = pair_feats

                recycle_bins, plddt_max = return_predcadist_plddt(Predxyz, Plddt)

        return Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, seq_feats, pair_feats


############# Fitness Module #############
class PretrainGAT(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim):
        super().__init__()

        self.node_dim = node_dim
        self.n_head = n_head
        self.head_dim = node_dim // n_head

        # * to alpha
        self.query = torch.nn.Linear(node_dim, self.head_dim * n_head, bias=False)
        self.key = torch.nn.Linear(node_dim, self.head_dim * n_head, bias=False)
        self.value = torch.nn.Linear(node_dim, self.head_dim * n_head, bias=False)
        self.pair2alpha = torch.nn.Linear(pair_dim, n_head, bias=False)
        self.conv2dalpha = torch.nn.Sequential(torch.nn.InstanceNorm2d(n_head * 2), torch.nn.Conv2d(n_head * 2, n_head, 3, 1, 1), torch.nn.LeakyReLU())
        self.alpha_mask_plddt = torch.nn.Sequential(torch.nn.InstanceNorm2d(n_head + 1), torch.nn.Conv2d(n_head + 1, n_head, 3, 1, 1), torch.nn.LeakyReLU())

        # output
        self.out_transform = torch.nn.Sequential(torch.nn.LayerNorm(n_head * pair_dim + node_dim), torch.nn.Linear(n_head * pair_dim + node_dim, node_dim * 2), torch.nn.LayerNorm(node_dim * 2), torch.nn.LeakyReLU(), torch.nn.Linear(node_dim * 2, node_dim))
        self.layer_norm = torch.nn.LayerNorm(node_dim)
        self.alpha2pair = torch.nn.Sequential(torch.nn.InstanceNorm2d(n_head + pair_dim), torch.nn.Conv2d(n_head + pair_dim, pair_dim, 3, 1, 1), torch.nn.LeakyReLU())

    @staticmethod
    def _heads(x, n_head, n_ch):

        # x = [..., n_head * n_ch] -> [..., n_head, n_ch]
        s = list(x.shape)[:-1] + [n_head, n_ch]
        return x.view(*s)

    def _node2alpha(self, x):
        query_l = self._heads(self.query(x), self.n_head, self.head_dim)
        key_l = self._heads(self.key(x), self.n_head, self.head_dim)

        # query_l = [N, L, n_head, head_dim]
        # key_l = [N, L, n_head, head_dim]

        query_l = query_l.permute(0, 2, 1, 3)
        key_l = key_l.permute(0, 2, 3, 1)

        # query_l = [N, n_head, L, head_dim]
        # key_l = [N, n_head, head_dim, L]

        alpha = torch.matmul(query_l, key_l) / torch.sqrt(torch.FloatTensor([self.head_dim]).to(x.device))
        alpha = alpha.permute(0, 2, 3, 1)
        return alpha

    def _pair2alpha(self, z):
        alpha = self.pair2alpha(z)
        return alpha

    def _node_aggregation(self, alpha, x):
        N = x.shape[0]
        value_l = self._heads(self.value(x), self.n_head, self.head_dim)

        # value_l = [N, L, n_head, head_dim]

        value_l = value_l.permute(0, 2, 1, 3)

        # value_l = [N, n_head, L, head_dim]

        x = torch.matmul(alpha.permute(0, 3, 1, 2), value_l)

        # x = [N, n_head, L, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [N, L, n_head, head_dim]

        x = x.view(N, -1, self.node_dim)

        # x = [N, L, node_dim]

        return x

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        node_from_pair = alpha.unsqueeze(-1) * z.unsqueeze(-2)

        # node_from_pair = [N, L, L, n_head, pair_dim]

        node_from_pair = node_from_pair.sum(dim=2).reshape(N, L, -1)

        # node_from_pair = [N, L, n_head * pair_dim]

        return node_from_pair

    def forward(self, x, z, plddt):

        # x = [N, L, node_dim]
        # z = [N, L, L, pair_dim]
        # plddt = [N, L, L, 1]

        alpha_from_node = self._node2alpha(x)
        alpha_from_pair = self._pair2alpha(z)

        # alpha_from_node = [N, L, L, n_head]
        # alpha_from_pair = [N, L, L, n_head]

        alpha = self.conv2dalpha(torch.cat((alpha_from_pair, alpha_from_node), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        alpha = self.alpha_mask_plddt(torch.cat((alpha, plddt), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # alpha = [N, L, L, n_head]

        node_from_node = self._node_aggregation(alpha, x)
        node_from_pair = self._pair_aggregation(alpha, z)

        # node_from_node = [N, L, node_dim]
        # node_from_pair = [N, L, n_head * pair_dim]

        x_out = self.out_transform(torch.cat([node_from_pair, node_from_node], dim=-1))
        x = self.layer_norm(x + x_out)
        return x, self.alpha2pair(torch.cat((z, alpha), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class PretrainEncoder(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim, num_layer):
        super().__init__()

        self.esm2_transform = torch.nn.Sequential(torch.nn.LayerNorm(1280), torch.nn.Linear(1280, 640), torch.nn.LeakyReLU(), torch.nn.Linear(640, 320), torch.nn.LeakyReLU(), torch.nn.Linear(320, node_dim), torch.nn.LeakyReLU(), torch.nn.Linear(node_dim, node_dim))
        self.pair_encoder = torch.nn.Linear(3, pair_dim)
        self.blocks = torch.nn.ModuleList([PretrainGAT(node_dim, n_head, pair_dim) for _ in range(num_layer)])
        self.pair_transform = torch.nn.Sequential(torch.nn.LayerNorm(pair_dim), torch.nn.Linear(pair_dim, pair_dim * 2), torch.nn.LeakyReLU(), torch.nn.Linear(pair_dim * 2, pair_dim), torch.nn.LeakyReLU())

    def forward(self, embedding, pair, plddt):

        # plddt = [N, L, L]
        # embedding = [N, L, 1280]
        # pair = [N, L, L, 128]

        embedding = self.esm2_transform(embedding)

        # embedding = [N, L, node_dim]

        pair = self.pair_encoder(pair)

        # pair = [N, L, L, pair_dim]

        for block in self.blocks:
            embedding, pair = block(embedding, pair, plddt.unsqueeze(-1))

        pair = self.pair_transform(pair)

        return embedding, pair


class PretrainModel(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim, num_layer):
        super().__init__()

        self.pretrain_encoder = PretrainEncoder(node_dim, n_head, pair_dim, num_layer)
        self.pretrain_single_mlp = torch.nn.Linear(node_dim, 20)
        self.pretrain_double_mlp = torch.nn.Sequential(torch.nn.InstanceNorm2d(pair_dim), torch.nn.Conv2d(pair_dim, 200, 3, 1, 1), torch.nn.LeakyReLU(), torch.nn.InstanceNorm2d(200), torch.nn.Conv2d(200, 400, 3, 1, 1), torch.nn.LeakyReLU())
        self.logits_coef1 = torch.nn.Parameter(torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], requires_grad=True))
        self.logits_coef2 = torch.nn.Parameter(torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], requires_grad=True))

    def forward(self, data):

        single_feat, double_feat = self.pretrain_encoder(data["1d"], data["3d"], data["plddt"])

        single_logits = self.pretrain_single_mlp(single_feat)

        single_logits = self.logits_coef1[None, :, None, None] * torch.cat((single_logits.unsqueeze(1), data["single_logits"]), dim=1)
        single_logits = single_logits.mean(1)

        double_logits = self.pretrain_double_mlp(double_feat.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        double_logits = self.logits_coef2[None, :, None, None, None] * torch.cat((double_logits.unsqueeze(1), data["double_logits"]), dim=1)
        double_logits = double_logits.mean(1)
        return single_logits, double_logits


################ SPIRED-Fitness ######################


class SPIRED_Fitness_Union(torch.nn.Module):
    def __init__(self, device_list):
        super().__init__()
        self.SPIRED = SPIRED_Model(depth=2, channel=128, device_list=device_list)
        self.Fitness = PretrainModel(node_dim=32, num_layer=2, n_head=8, pair_dim=32).to(device_list[-1])

    def forward(self, true_aa, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits):
        Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, seq_feats, pair_feats = self.SPIRED(true_aa, f1d_esm2_3B, no_recycles=1)
        ## feature for Fitness Module
        xyz_LL3 = Predxyz["4th"][-1].permute(0, 2, 3, 1).contiguous()
        plddt_value = Plddt["4th"][-1]

        f1d_esm2_650M, plddt_value, esm1v_single_logits, esm1v_double_logits = (f1d_esm2_650M.to(xyz_LL3.device), plddt_value.to(xyz_LL3.device), esm1v_single_logits.to(xyz_LL3.device), esm1v_double_logits.to(xyz_LL3.device))

        pred_fitness_logits, double_logits = self.Fitness({"1d": f1d_esm2_650M, "3d": xyz_LL3, "plddt": plddt_value, "single_logits": esm1v_single_logits, "double_logits": esm1v_double_logits})
        return pred_fitness_logits, double_logits, Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, seq_feats, pair_feats


################ SPIRED-Stab ######################


class GAT(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim):
        super().__init__()

        self.node_dim = node_dim
        self.n_head = n_head
        self.head_dim = node_dim // n_head

        # * to alpha
        self.query = torch.nn.Linear(node_dim, self.head_dim * n_head, bias=False)
        self.key = torch.nn.Linear(node_dim, self.head_dim * n_head, bias=False)
        self.value = torch.nn.Linear(node_dim, self.head_dim * n_head, bias=False)
        self.pair2alpha = torch.nn.Linear(pair_dim, n_head, bias=False)
        self.conv2dalpha = torch.nn.Sequential(torch.nn.InstanceNorm2d(n_head * 2), torch.nn.Conv2d(n_head * 2, n_head, 3, 1, 1), torch.nn.LeakyReLU())
        self.alpha_mask_plddt = torch.nn.Sequential(torch.nn.InstanceNorm2d(n_head + 1), torch.nn.Conv2d(n_head + 1, n_head, 3, 1, 1), torch.nn.LeakyReLU())

        # output
        self.out_transform = torch.nn.Sequential(torch.nn.LayerNorm(n_head * pair_dim + node_dim), torch.nn.Linear(n_head * pair_dim + node_dim, node_dim * 2), torch.nn.LayerNorm(node_dim * 2), torch.nn.LeakyReLU(), torch.nn.Linear(node_dim * 2, node_dim))
        self.layer_norm = torch.nn.LayerNorm(node_dim)
        self.alpha2pair = torch.nn.Sequential(torch.nn.InstanceNorm2d(n_head + pair_dim), torch.nn.Conv2d(n_head + pair_dim, pair_dim, 3, 1, 1), torch.nn.LeakyReLU())

    @staticmethod
    def _heads(x, n_head, n_ch):

        # x = [..., n_head * n_ch] -> [..., n_head, n_ch]
        s = list(x.shape)[:-1] + [n_head, n_ch]
        return x.view(*s)

    def _node2alpha(self, x):
        query_l = self._heads(self.query(x), self.n_head, self.head_dim)
        key_l = self._heads(self.key(x), self.n_head, self.head_dim)

        # query_l = [N, L, n_head, head_dim]
        # key_l = [N, L, n_head, head_dim]

        query_l = query_l.permute(0, 2, 1, 3)
        key_l = key_l.permute(0, 2, 3, 1)

        # query_l = [N, n_head, L, head_dim]
        # key_l = [N, n_head, head_dim, L]

        alpha = torch.matmul(query_l, key_l) / torch.sqrt(torch.FloatTensor([self.head_dim]).to(x.device))
        alpha = alpha.permute(0, 2, 3, 1)
        return alpha

    def _pair2alpha(self, z):
        alpha = self.pair2alpha(z)
        return alpha

    def _node_aggregation(self, alpha, x):
        N = x.shape[0]
        value_l = self._heads(self.value(x), self.n_head, self.head_dim)

        # value_l = [N, L, n_head, head_dim]

        value_l = value_l.permute(0, 2, 1, 3)

        # value_l = [N, n_head, L, head_dim]

        x = torch.matmul(alpha.permute(0, 3, 1, 2), value_l)

        # x = [N, n_head, L, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [N, L, n_head, head_dim]

        x = x.view(N, -1, self.node_dim)

        # x = [N, L, node_dim]

        return x

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        node_from_pair = alpha.unsqueeze(-1) * z.unsqueeze(-2)

        # node_from_pair = [N, L, L, n_head, pair_dim]

        node_from_pair = node_from_pair.sum(dim=2).reshape(N, L, -1)

        # node_from_pair = [N, L, n_head * pair_dim]

        return node_from_pair

    def forward(self, x, z, plddt):

        # x = [N, L, node_dim]
        # z = [N, L, L, pair_dim]
        # plddt = [N, L, L, 1]

        alpha_from_node = self._node2alpha(x)
        alpha_from_pair = self._pair2alpha(z)

        # alpha_from_node = [N, L, L, n_head]
        # alpha_from_pair = [N, L, L, n_head]

        alpha = self.conv2dalpha(torch.cat((alpha_from_pair, alpha_from_node), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        alpha = self.alpha_mask_plddt(torch.cat((alpha, plddt), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # alpha = [N, L, L, n_head]

        node_from_node = self._node_aggregation(alpha, x)
        node_from_pair = self._pair_aggregation(alpha, z)

        # node_from_node = [N, L, node_dim]
        # node_from_pair = [N, L, n_head * pair_dim]

        x_out = self.out_transform(torch.cat([node_from_pair, node_from_node], dim=-1))
        x = self.layer_norm(x + x_out)
        return x, self.alpha2pair(torch.cat((z, alpha), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class Model(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim, num_layer):
        super().__init__()

        self.esm2_transform = torch.nn.Sequential(torch.nn.LayerNorm(1280), torch.nn.Linear(1280, 640), torch.nn.LeakyReLU(), torch.nn.Linear(640, 320), torch.nn.LeakyReLU(), torch.nn.Linear(320, node_dim), torch.nn.LeakyReLU(), torch.nn.Linear(node_dim, node_dim))
        self.pair_encoder = torch.nn.Linear(3, pair_dim)
        self.blocks = torch.nn.ModuleList([GAT(node_dim, n_head, pair_dim) for _ in range(num_layer)])
        self.mlp = torch.nn.Sequential(torch.nn.LayerNorm(node_dim), torch.nn.Linear(node_dim, node_dim // 2), torch.nn.LeakyReLU(), torch.nn.Linear(node_dim // 2, 1))
        self.mlp_for_dTm = torch.nn.Sequential(torch.nn.LayerNorm(node_dim), torch.nn.Linear(node_dim, node_dim // 2), torch.nn.LeakyReLU(), torch.nn.Linear(node_dim // 2, 1))
        self.finetune_ddG_coef = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.finetune_dTm_coef = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))

    def forward(self, wt_data, mut_data, mut_pos):

        wt_embedding = self.esm2_transform(wt_data["embedding"])
        mut_embedding = self.esm2_transform(mut_data["embedding"])

        # embedding = [N, L, node_dim]

        wt_pair = self.pair_encoder(wt_data["pair"])
        mut_pair = self.pair_encoder(mut_data["pair"])

        # pair = [N, L, L, pair_dim]

        for block in self.blocks:
            wt_embedding, wt_pair = block(wt_embedding, wt_pair, wt_data["plddt"].unsqueeze(-1))
            mut_embedding, mut_pair = block(mut_embedding, mut_pair, mut_data["plddt"].unsqueeze(-1))

        mut_dG = self.mlp((mut_embedding * mut_pos.unsqueeze(-1)).sum(1)).squeeze(-1)
        wt_dG = self.mlp((wt_embedding * mut_pos.unsqueeze(-1)).sum(1)).squeeze(-1)
        mut_dG = mut_dG * self.finetune_ddG_coef
        wt_dG = wt_dG * self.finetune_ddG_coef

        mut_Tm = self.mlp_for_dTm((mut_embedding * mut_pos.unsqueeze(-1)).sum(1)).squeeze(-1)
        wt_Tm = self.mlp_for_dTm((wt_embedding * mut_pos.unsqueeze(-1)).sum(1)).squeeze(-1)
        mut_Tm = mut_Tm * self.finetune_dTm_coef
        wt_Tm = wt_Tm * self.finetune_dTm_coef
        return mut_dG - wt_dG, mut_Tm - wt_Tm


class SPIRED_Stab(torch.nn.Module):
    def __init__(self, device_list):
        super().__init__()
        self.SPIRED = SPIRED_Model(depth=2, channel=128, device_list=device_list)
        self.Stab = Model(node_dim=32, num_layer=3, n_head=8, pair_dim=64).to(device_list[-1])

    def forward(self, wt_data, mut_data, mut_pos_torch_list):

        # wt data
        wt_Predxyz, PredCadistavg, wt_Plddt, ca, cb, omega, theta, phi, wt_phi_psi_1D, seq_feats, pair_feats = self.SPIRED(wt_data["target_tokens"], wt_data["esm2-3B"], no_recycles=1)
        wt_features = {"Predxyz": wt_Predxyz, "Plddt": wt_Plddt, "phi_psi_1D": wt_phi_psi_1D}
        wt_data["pair"] = wt_Predxyz["4th"][-1].permute(0, 2, 3, 1).contiguous()
        wt_data["plddt"] = wt_Plddt["4th"][-1]
        wt_data["embedding"] = wt_data["embedding"].to(wt_data["pair"].device)

        # mut data
        mut_Predxyz, PredCadistavg, mut_Plddt, ca, cb, omega, theta, phi, mut_phi_psi_1D, seq_feats, pair_feats = self.SPIRED(mut_data["target_tokens"], mut_data["esm2-3B"], no_recycles=1)
        mut_features = {"Predxyz": mut_Predxyz, "Plddt": mut_Plddt, "phi_psi_1D": mut_phi_psi_1D}
        mut_data["pair"] = mut_Predxyz["4th"][-1].permute(0, 2, 3, 1).contiguous()
        mut_data["plddt"] = mut_Plddt["4th"][-1]
        mut_data["embedding"] = mut_data["embedding"].to(mut_data["pair"].device)

        ddG, dTm = self.Stab(wt_data, mut_data, mut_pos_torch_list)
        return ddG, dTm, wt_features, mut_features
