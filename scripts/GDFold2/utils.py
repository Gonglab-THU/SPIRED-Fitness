import os
import json
import torch
import numpy as np

int2AA = {0: "GLY", 1: "ALA", 2: "CYS", 3: "GLU", 4: "ASP", 5: "PHE", 6: "ILE", 7: "HIS", 8: "LYS", 9: "MET", 10: "LEU", 11: "ASN", 12: "GLN", 13: "PRO", 14: "SER", 15: "ARG", 16: "THR", 17: "TRP", 18: "VAL", 19: "TYR"}
aa2int = {"G": 0, "A": 1, "C": 2, "E": 3, "D": 4, "F": 5, "I": 6, "H": 7, "K": 8, "M": 9, "L": 10, "N": 11, "Q": 12, "P": 13, "S": 14, "R": 15, "T": 16, "W": 17, "V": 18, "Y": 19}


def fasta2seq(fasta):
    with open(fasta, "r") as f:
        seq = list("".join(line.strip() for line in f if not line.startswith(">")))
    return [aa2int[i] for i in seq]


def get_pred(pred, mode="Cerebra"):
    data = np.load(pred)
    trans = lambda x: torch.FloatTensor(x)
    if mode == "Cerebra":
        info = {"reference": list(data["reference"]), "rotation": trans(data["rotation"]), "translation": trans(data["translation"]), "dihedrals": trans(data["dihedrals"]), "plddt": trans(data["plddt"])}
    if mode == "SPIRED":
        info = {"reference": list(data["reference"]), "translation": trans(data["translation"]), "dihedrals": trans(data["dihedrals"]), "plddt": trans(data["plddt"])}
    if mode == "Rosetta":
        info = {"dist": trans(data["dist"]).permute(2, 0, 1), "omega": trans(data["omega"]).permute(2, 0, 1), "theta": trans(data["theta"]).permute(2, 0, 1), "phi": trans(data["phi"]).permute(2, 0, 1)}
    return info


def get_params(seq):
    params = {}
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    with open(f"{scriptdir}/params.json") as jsonfile:
        statistic = json.load(jsonfile)
    for key in statistic.keys():
        params[key] = torch.FloatTensor(np.array(statistic[key]))
    params["vdw_mask"] = torch.IntTensor(vdw_mask(seq))
    params["vdw_dist"] = vdw_dist(seq, params["vdw_radius"])
    return params


def vdw_mask(seq):
    length = len(seq) * 7
    mask = np.ones((length, length))
    mask -= np.diag(mask.diagonal())
    gly_index = [i for i, x in enumerate(seq) if x == 0]
    pro_index = [i for i, x in enumerate(seq) if x == 13]
    for i in range(length):
        if i % 7 == 0:
            mask[i : i + 5, i : i + 5] = 0
        if (i - 1) % 7 == 0 and (i + 8) < length:
            mask[i, i + 8] = 0
        if (i - 1) % 7 == 0:
            mask[i, i + 4] = 0
        if (i - 2) % 7 == 0:
            mask[i, i + 4] = 0
    for gly in gly_index:
        mask[gly * 7 + 3] = 0
        mask[:, gly * 7 + 3] = 0
    for pro in pro_index:
        mask[pro * 7 + 6] = 0
        mask[:, pro * 7 + 6] = 0
    mask = np.triu(mask)
    mask += mask.T
    return mask


def vdw_dist(seq, vdw_radius):
    vdw_radius = vdw_radius.repeat(len(seq))
    vdw_dist = vdw_radius.unsqueeze(0) + vdw_radius.unsqueeze(1) - 1.2
    return vdw_dist


def output(path, name, seq, coords):
    [CA, C, N, CB, CEN, O, H] = [coords[..., i] for i in range(7)]
    all_coords = np.stack([N, CA, C, O, CB, CEN, H], axis=-2)
    symbols = ["N", "CA", "C", "O", "CB", "CEN", "H"]
    elements = ["N", "C", "C", "O", "C", "X", "H"]
    for n in range(coords.shape[0]):
        file_name = f"{path}/fold_{n}.pdb"
        with open(file_name, "w") as f:
            natom, chain = 1, "A"
            for l in range(len(seq)):
                for a in range(4):
                    f.write("{:<6}{:>5} {:^4} {:<3} {:>1}{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}{:>12}\n".format("ATOM", int(natom), symbols[a], int2AA.get(seq[l]), chain, l + 1, all_coords[n, l, a, 0], all_coords[n, l, a, 1], all_coords[n, l, a, 2], 1.00, 0.00, elements[a]))
                    natom += 1
                if int2AA.get(seq[l]) != "GLY":
                    for a in range(4, 6):
                        f.write("{:<6}{:>5} {:^4} {:<3} {:>1}{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}{:>12}\n".format("ATOM", int(natom), symbols[a], int2AA.get(seq[l]), chain, l + 1, all_coords[n, l, a, 0], all_coords[n, l, a, 1], all_coords[n, l, a, 2], 1.00, 0.00, elements[a]))
                        natom += 1
                else:
                    f.write("{:<6}{:>5} {:^4} {:<3} {:>1}{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}{:>12}\n".format("ATOM", int(natom), symbols[5], int2AA.get(seq[l]), chain, l + 1, all_coords[n, l, 5, 0], all_coords[n, l, 5, 1], all_coords[n, l, 5, 2], 1.00, 0.00, elements[5]))
                    natom += 1
                if int2AA.get(seq[l]) != "PRO":
                    f.write("{:<6}{:>5} {:^4} {:<3} {:>1}{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}{:>12}\n".format("ATOM", int(natom), symbols[-1], int2AA.get(seq[l]), chain, l + 1, all_coords[n, l, -1, 0], all_coords[n, l, -1, 1], all_coords[n, l, -1, 2], 1.00, 0.00, elements[-1]))
                    natom += 1
                else:
                    continue
