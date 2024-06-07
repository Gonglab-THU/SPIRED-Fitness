import torch
import torch.nn as nn
import Module
import time
from contextlib import ExitStack
from dataclasses import dataclass
from esmfold_openfold.tri_self_attn_block import TriangularSelfAttentionBlock
from utils_train_valid import return_predcadist_plddt


class RelativePosition(nn.Module):
    def __init__(self, bins, pairwise_state_dim):
        super().__init__()
        self.bins = bins
        # refer to ESMFold
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


####################################################################

eps = 1e-8


def getCadistAvg(relativepredxyz):
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
    def __init__(self, depth=2, channel=128):
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

        ### Triangular_SelfAttention_Block
        seq_feats, pair_feats = trunk_iter(seq_feats, pair_feats, residx, mask)
        residual, pair_feats = self.ircn0(pair_feats)

        ### Structure Prediction
        #### predict coordinate and CA distance first time
        predxyz, pair_feats = self.predDxyz0(pair_feats)
        predxyz = predxyz * maskdiag
        pair_feats = torch.cat((residual, pair_feats), dim=-1)
        residual, pair_feats = self.ircn1(pair_feats)

        relativepredxyz = predxyz[:, :, :, None, :] - predxyz[:, :, :, :, None]  # (N,3,L,L,L)
        cadistavg = getCadistAvg(relativepredxyz)

        Predxyz.append(predxyz)
        PredCadistavg.append(cadistavg)
        Plddt.append(self.plddt0(pair_feats, cadistavg))

        #### predict coordinate (change) and CA distance second time
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
    def __init__(self, depth=2, channel=128):
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

        ### Triangular_SelfAttention_Block
        seq_feats, pair_feats = trunk_iter(seq_feats, pair_feats, residx, mask)
        residual, pair_feats = self.ircn0(pair_feats)

        ### Structure Prediction
        #### predict coordinate and CA distance first time
        predxyz, pair_feats = self.predDxyz0(pair_feats)
        predxyz = predxyz * maskdiag
        pair_feats = torch.cat((residual, pair_feats), dim=-1)
        residual, pair_feats = self.ircn1(pair_feats)

        relativepredxyz = predxyz[:, :, :, None, :] - predxyz[:, :, :, :, None]  # (N,3,L,L,L)
        cadistavg = getCadistAvg(relativepredxyz)

        Predxyz.append(predxyz)
        PredCadistavg.append(cadistavg)
        Plddt.append(self.plddt0(pair_feats, cadistavg))

        #### predict coordinate (change) and CA distance second time
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
    def __init__(self, depth=2, channel=128):
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

        seq_feats, pair_feats = trunk_iter(seq_feats, pair_feats, residx, mask)
        residual, pair_feats = self.ircn0(pair_feats)

        ### Structure Prediction
        #### predict coordinate and CA distance first time
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

        #### Predict Ca coordinates third time
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

        #### Predict Ca coordinates fourth time
        dxyz, pair_feats = self.predDxyz0(pair_feats, cadistavg)
        dxyz = dxyz * maskdiag
        # pair_feats += residual
        pair_feats = torch.cat((residual, pair_feats), dim=-1)
        residual, pair_feats = self.ircn4(pair_feats)

        predxyz = predxyz + dxyz
        relativepredxyz = predxyz[:, :, :, None, :] - predxyz[:, :, :, :, None]  # (N,3,L,L,L)
        cadistavg = getCadistAvg(relativepredxyz)

        Predxyz.append(predxyz)
        PredCadistavg.append(cadistavg)
        Plddt.append(self.plddt3(pair_feats, cadistavg))

        #### Predict Ca coordinates fifth time
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

        #### Predict Ca coordinates sixth time
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
    def __init__(self, depth=2, channel=128, device_list=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]):
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

        ######
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

    def forward(self, true_aa, f1d, no_recycles):
        s1 = time.time()
        Predxyz = {}
        PredCadistavg = {}
        Plddt = {}

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

                ### FU1
                Predxyz0, PredCadistavg0, Plddt0, seq_feats, pair_feats = self.block0(seq_feats, pair_feats, true_aa, residx, mask, maskdiag)
                Predxyz["1st"], PredCadistavg["1st"], Plddt["1st"] = Predxyz0, PredCadistavg0, Plddt0
                seq_feats, pair_feats, featureResidual, maskdiag, residx, mask, true_aa = (seq_feats.to(self.device1), pair_feats.to(self.device1), pair_feats.to(self.device1), maskdiag.to(self.device1), residx.to(self.device1), mask.to(self.device1), true_aa.to(self.device1))
                pair_feats = (pair_feats + pair_feats.permute(0, 2, 1, 3)) / 2.0

                ### FU2
                Predxyz0, PredCadistavg0, Plddt0, seq_feats, pair_feats = self.block1(seq_feats, pair_feats, true_aa, residx, mask, maskdiag)
                Predxyz["2nd"], PredCadistavg["2nd"], Plddt["2nd"] = Predxyz0, PredCadistavg0, Plddt0
                pair_feats = (pair_feats + pair_feats.permute(0, 2, 1, 3)) / 2.0
                feature = torch.cat((featureResidual, pair_feats), dim=-1)
                featureResidual, pair_feats = self.featureReduction0(feature)
                seq_feats, pair_feats, featureResidual, maskdiag, residx, mask, true_aa = (seq_feats.to(self.device2), pair_feats.to(self.device2), featureResidual.to(self.device2), maskdiag.to(self.device2), residx.to(self.device2), mask.to(self.device2), true_aa.to(self.device2))

                ### FU3
                Predxyz0, PredCadistavg0, Plddt0, seq_feats, pair_feats = self.block2(seq_feats, pair_feats, true_aa, residx, mask, maskdiag)
                Predxyz["3rd"], PredCadistavg["3rd"], Plddt["3rd"] = Predxyz0, PredCadistavg0, Plddt0
                pair_feats = (pair_feats + pair_feats.permute(0, 2, 1, 3)) / 2.0
                feature = torch.cat((featureResidual, pair_feats), dim=-1)
                featureResidual, pair_feats = self.featureReduction1(feature)
                seq_feats, pair_feats, featureResidual, maskdiag, residx, mask, true_aa = (seq_feats.to(self.device3), pair_feats.to(self.device3), featureResidual.to(self.device3), maskdiag.to(self.device3), residx.to(self.device3), mask.to(self.device3), true_aa.to(self.device3))

                ### FU4
                Predxyz0, PredCadistavg0, Plddt0, seq_feats, pair_feats = self.block3(seq_feats, pair_feats, true_aa, residx, mask, maskdiag)
                Predxyz["4th"], PredCadistavg["4th"], Plddt["4th"] = Predxyz0, PredCadistavg0, Plddt0
                pair_feats = (pair_feats + pair_feats.permute(0, 2, 1, 3)) / 2.0
                feature = torch.cat((featureResidual, pair_feats), dim=-1)
                featureResidual, pair_feats = self.featureReduction2(feature)

                ca, cb, omega, theta, phi = self.toCACBangle(pair_feats)
                phi_psi_1D = self.to_phi_psi(seq_feats)  # (batch, 2, L-1)

                recycle_s = seq_feats
                recycle_z = pair_feats

                recycle_bins, plddt_max = return_predcadist_plddt(Predxyz, Plddt)

        return Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, seq_feats, pair_feats
