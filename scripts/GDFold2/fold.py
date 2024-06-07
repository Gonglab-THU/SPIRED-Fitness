import math
import timeit
import numpy as np
import torch
from Bio import SeqIO
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils import *
from core import GradientDescent


def get_args():
    parser = argparse.ArgumentParser(description="GDFold2: a fast and parallelizable protein folding environment")

    parser.add_argument("fasta", type=str, help="input protein sequence (.fasta format)")
    parser.add_argument("output", type=str, help="output directory name")

    parser.add_argument("-n", dest="npose", type=int, default=1, help="number of structures to predict simultaneously, default=1")
    parser.add_argument("-s", dest="steps", type=int, default=400, help="number of optimization steps, default=400")
    parser.add_argument("-d", dest="device", type=str, default="cpu", help="device to run the task, default=cpu")
    args = parser.parse_args()
    return args


class Cerebra(GradientDescent):
    """
    Protein folding environment for Cerebra.
    """

    def _vector_term(self, mat, coords):
        super()._vector_term(mat, coords)
        return self.vector_loss.mean()

    def _dihedral_term(self, mat, coords):
        super()._dihedral_term(mat, coords)
        return self.dihedral_loss.mean()


class SPIRED(GradientDescent):
    """
    Protein folding environment for Spried.
    """

    def _vector_term(self, mat, coords):
        loss = nn.MSELoss(reduction="none")
        reference = self.pred["reference"][: self.npose]
        translation = self.pred["translation"][: self.npose].to(self.device)
        index = list(torch.arange(self.npose))
        CA_ref = coords[0][index, reference]
        coord_pred = coords[0] - CA_ref.unsqueeze(1)
        coord_label = translation.to(self.device)
        self.vectors_loss = torch.sum(loss(coord_pred, coord_label), dim=-1)
        return self.vectors_loss.mean()


class Rosetta(GradientDescent):
    """
    Protein folding environment for RoseTTAFold/trRosetta.

    From https://github.com/RosettaCommons/RoseTTAFold/blob/main/network/trFold.py
    """

    def _reshape(self):
        super()._reshape()
        self.dcut = 19.5
        self.alpha = 1.57
        self.astep = np.deg2rad(10.0)
        self.dstep = 0.5
        self.pcut = 0.5
        self.clash = 2.0
        self.weight = 2.0
        self.sg = torch.Tensor([[[-0.0909, 0.0606, 0.1688, 0.2338, 0.2554, 0.2338, 0.1688, 0.0606, -0.0909]]]).float().to(self.device)
        self._process_cst()

    def _get_dih(self, a, b, c, d):
        v0 = a - b
        v1 = c - b
        v2 = d - c
        v1 = F.normalize(v1, dim=-1)
        v = v0 - torch.sum(v0 * v1, dim=-1, keepdim=True) * v1
        w = v2 - torch.sum(v2 * v1, dim=-1, keepdim=True) * v1
        x = torch.sum(v * w, dim=-1)
        y = torch.sum(torch.cross(v1, v, dim=-1) * w, dim=-1)
        return torch.atan2(y, x)

    def _get_ang(self, a, b, c):
        v = F.normalize(a - b, dim=-1)
        w = F.normalize(c - b, dim=-1)
        vw = torch.sum(v * w, dim=-1)
        vw = torch.clamp(vw, min=-1.0 + self.epsilon, max=1.0 - self.epsilon)
        return torch.acos(vw)

    def _akima(self, y, h):
        m = (y[:, 1:] - y[:, :-1]) / h
        m4m3 = torch.abs(m[:, 3:] - m[:, 2:-1])
        m2m1 = torch.abs(m[:, 1:-2] - m[:, :-3])
        t = torch.nan_to_num((m4m3 * m[:, 1:-2] + m2m1 * m[:, 2:-1]) / (m4m3 + m2m1))
        dy = y[:, 3:-2] - y[:, 2:-3]
        coef = torch.stack([y[:, 2:-3], t[:, :-1], (3 * dy / h - 2 * t[:, :-1] - t[:, 1:]) / h, (t[:, :-1] + t[:, 1:] - 2 * dy / h) / h**2], dim=-1)
        return coef

    def _process_cst(self):
        # dfire background correction for distograms
        self.bkgd = (torch.linspace(4.25, 19.75, 32, device=self.device) / self.pcut) ** self.alpha

        # background correction for phi
        ang = torch.linspace(0, math.pi, 19, device=self.device)[:-1]
        self.bkgp = 0.5 * (torch.cos(ang) - torch.cos(ang + self.astep))

        # paddings for distograms:
        # left - linear clash; right - zeroes
        padRsize = self.sg.shape[-1] // 2 + 3
        padLsize = padRsize + 8
        padR = torch.zeros(padRsize, device=self.device)
        padL = torch.arange(1, padLsize + 1, device=self.device).flip(0) * self.clash
        self.padR = padR[:, None]
        self.padL = padL[:, None]

        # read and reshape predictions
        pred = lambda x: self.pred[x].to(self.device) + self.epsilon
        p_dist, p_omega, p_theta, p_phi = map(pred, ("dist", "omega", "theta", "phi"))
        p_dist = p_dist / torch.sum(p_dist, dim=0)[None]
        p_omega = p_omega / torch.sum(p_omega, dim=0)[None]
        p_theta = p_theta / torch.sum(p_theta, dim=0)[None]
        p_phi = p_phi / torch.sum(p_phi, dim=0)[None]
        p20 = 1.0 - (p_dist[-1] + p_omega[-1] + p_theta[-1] + p_phi[-1] + p_theta[-1].T + p_phi[-1].T) / 6
        i, j = torch.triu_indices(self.length, self.length, 1).to(self.device)
        sel = torch.where(p20[i, j] > self.pcut)[0]

        # indices for dist and omega (symmetric)
        self.i_s, self.j_s = i[sel], j[sel]
        # indices for theta and phi (asymmetric)
        self.i_a, self.j_a = torch.hstack([self.i_s, self.j_s]), torch.hstack([self.j_s, self.i_s])

        # background-corrected initial restraints
        cstd = -torch.log(p_dist[4:36, self.i_s, self.j_s] / self.bkgd[:, None])
        csto = -torch.log(p_omega[0:36, self.i_s, self.j_s] / (1.0 / 36))
        cstt = -torch.log(p_theta[0:36, self.i_a, self.j_a] / (1.0 / 36))
        cstp = -torch.log(p_phi[0:18, self.i_a, self.j_a] / self.bkgp[:, None])

        # padded restraints
        pad = self.sg.shape[-1] // 2 + 3
        cstd = torch.cat([self.padL + cstd[0], cstd, self.padR + cstd[-1]], dim=0)
        csto = torch.cat([csto[-pad:], csto, csto[:pad]], dim=0)
        cstt = torch.cat([cstt[-pad:], cstt, cstt[:pad]], dim=0)
        cstp = torch.cat([cstp[:pad].flip(0), cstp, cstp[-pad:].flip(0)], dim=0)

        # smoothed restraints
        cstd, csto, cstt, cstp = [nn.functional.conv1d(cst.T.unsqueeze(1), self.sg)[:, 0] for cst in [cstd, csto, cstt, cstp]]
        # force distance restraints vanish at long distances
        cstd = cstd - cstd[:, -1][:, None]

        self.coefd = self._akima(cstd, self.dstep).detach()
        self.coefo = self._akima(csto, self.astep).detach()
        self.coeft = self._akima(cstt, self.astep).detach()
        self.coefp = self._akima(cstp, self.astep).detach()
        self.ko = torch.arange(self.i_s.shape[0]).repeat(self.npose).to(self.device)
        self.kt = torch.arange(self.i_a.shape[0]).repeat(self.npose).to(self.device)

    def _vector_term(self, mat, coords):
        CB = coords[3]
        dij = torch.linalg.norm(CB[:, self.i_s] - CB[:, self.j_s], dim=-1)
        b, k = torch.where(dij < 20.0)
        dk = dij[b, k]
        kbin = torch.ceil((dk - 0.25) / 0.5).long()
        dx = (dk - 0.25) % 0.5
        c = self.coefd[k, kbin]
        loss_dist = self.weight * (c[:, 0] + c[:, 1] * dx + c[:, 2] * dx**2 + c[:, 3] * dx**3)
        return torch.mean(loss_dist)

    def _dihedral_term(self, mat, coords):
        CA, C, N, CB = [coords[i] for i in range(4)]
        astep = self.astep
        omega = self._get_dih(CA[:, self.i_s], CB[:, self.i_s], CB[:, self.j_s], CA[:, self.j_s]) + math.pi
        theta = self._get_dih(N[:, self.i_a], CA[:, self.i_a], CB[:, self.i_a], CB[:, self.j_a]) + math.pi
        phi = self._get_ang(CA[:, self.i_a], CB[:, self.i_a], CB[:, self.j_a])

        # omega
        omega_bin = torch.ceil((omega.view(-1) - astep / 2) / astep).long()
        d_omega = (omega.view(-1) - astep / 2) % astep
        c_omega = self.coefo[self.ko, omega_bin]
        loss_omega = torch.mean((c_omega[:, 0] + c_omega[:, 1] * d_omega + c_omega[:, 2] * d_omega**2 + c_omega[:, 3] * d_omega**3).view(self.npose, -1), dim=1)
        # theta
        theta_bin = torch.ceil((theta.view(-1) - astep / 2) / astep).long()
        d_theta = (theta.view(-1) - astep / 2) % astep
        c_theta = self.coeft[self.kt, theta_bin]
        loss_theta = torch.mean((c_theta[:, 0] + c_theta[:, 1] * d_theta + c_theta[:, 2] * d_theta**2 + c_theta[:, 3] * d_theta**3).view(self.npose, -1), dim=1)
        # phi
        phi_bin = torch.ceil((phi.view(-1) - astep / 2) / astep).long()
        d_phi = (phi.view(-1) - astep / 2) % astep
        c_phi = self.coefp[self.kt, phi_bin]
        loss_phi = torch.mean((c_phi[:, 0] + c_phi[:, 1] * d_phi + c_phi[:, 2] * d_phi**2 + c_phi[:, 3] * d_phi**3).view(self.npose, -1), dim=1)
        loss_angle = self.weight * (loss_omega + loss_theta + loss_phi)
        return torch.mean(loss_angle)


def main():
    args = get_args()
    for index in SeqIO.parse(args.fasta, "fasta"):
        seq = [aa2int[i] for i in str(index.seq)]
        name = index.description
        pred = get_pred(f"{args.output}/{name}/GDFold2/input.npz", "SPIRED")
        params = get_params(seq)

        print("Initializing environment...")
        folder = globals()["SPIRED"](seq, pred, params, args.npose, args.steps, args.device)
        print("Folding...")
        coords = folder._fold()

        output(f"{args.output}/{name}/GDFold2", name, seq, coords)


if __name__ == "__main__":
    start_time = timeit.default_timer()
    main()
    end_time = timeit.default_timer()
    print("Running time: {:.2f}s".format(end_time - start_time))
