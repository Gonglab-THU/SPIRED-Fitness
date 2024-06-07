import numpy as np
import scipy
import torch, h5py
import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--sample_list_file", type=str, help="path of dir storing original asm files")
parser.add_argument("--pred_pdb_dir", type=str, default="", help="path of dir predicted struct pdb files")
parser.add_argument("--fasta_dir", type=str, default="", help="path of dir predicted struct pdb files")
parser.add_argument("--true_coords_hdf5", type=str, help="path of coords hdf5 file")
opt = parser.parse_args()


seq2token_dict = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19, "X": 20, "O": 21, "U": 22, "B": 23, "Z": 24, "-": 25, ".": 26}

token2seq_dict = {
    0: "A",
    1: "C",
    2: "D",
    3: "E",
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",
    21: "O",
    22: "U",
    23: "B",
    24: "Z",
    25: "-",
    26: ".",
    27: "<mask>",
    28: "<pad>",
}


def PsiPhi(atoms):
    eps = 1e-6

    def psi(CA, C, N):
        a = N[1:] - C[:-1]
        b = C - CA
        c = N - CA
        ab = torch.cross(a, b[:-1])
        bc = torch.cross(b[:-1], c[:-1])
        ca = torch.cross(c[:-1], a)

        cos_ca_b = torch.sum(ca * b[:-1], dim=-1) / (torch.linalg.norm(ca, dim=-1) * torch.linalg.norm(b[:-1], dim=-1) + eps)
        cospsi = torch.sum(ab * bc, dim=-1) / (torch.linalg.norm(ab, dim=-1) * torch.linalg.norm(bc, dim=-1) + eps)
        cospsi = np.pi - torch.arccos(torch.clamp(cospsi, max=1, min=-1))
        return (cos_ca_b / abs(cos_ca_b)) * cospsi

    def phi(CA, C, N):
        b = C - CA
        c = N - CA
        d = C[:-1] - N[1:]
        bc = torch.cross(b[1:], c[1:])
        cd = torch.cross(c[1:], d)
        bd = torch.cross(b[1:], d)
        cos_bd_c = torch.sum(bd * c[1:], dim=-1) / (torch.linalg.norm(bd, dim=-1) * torch.linalg.norm(c[1:], dim=-1) + eps)
        cosphi = torch.sum(bc * cd, dim=-1) / (torch.linalg.norm(bc, dim=-1) * torch.linalg.norm(cd, dim=-1) + eps)
        cosphi = np.pi - torch.arccos(torch.clamp(cosphi, max=1, min=-1))
        return (cos_bd_c / abs(cos_bd_c)) * cosphi

    N, CA, C = atoms[:, 0], atoms[:, 1], atoms[:, 2]
    return torch.stack([psi(CA, C, N), phi(CA, C, N)], dim=1)


def dist2bins(mat):
    """
    This function is for label
    bins definition: left open, right close. Because cutoff 8 definition: d<=8
    two kinds of bin length: 0.5,1.0
    bins:3.0],(3.0,3.5],(3.5,4.0],...,(21.5,22],(22,23],(23,24],...,(29,30],(30,
    Nbin=(22-3)*2+1+(30-22)+1=48

    mat: real value matrix, (L,L)

    return: (L,L,Nbins)

    """
    bins = list(np.arange(3.0, 22, 0.5)) + list(np.arange(22.0, 31.0, 1.0))
    L = mat.shape[0]
    Nbin = len(bins) + 1  # 48
    tensor = np.zeros([L, L, Nbin])
    tensor[..., 0] = (mat <= bins[0]).astype(int)
    for i in range(len(bins) - 1):
        tensor[..., i + 1] = ((mat > bins[i]) * (mat <= bins[i + 1])).astype(int)  # 任何情况 NaN 都对应纯0向量
    tensor[..., -1] = (mat > bins[-1]).astype(int)
    return tensor


def omega2bins(mat, distmat, dmax=30.0):
    """
    This function is for label
    bins definition: left open, right close
    anglebin=(beg0,end0]
    first bin=[-180,-165]
    Nbin=24+1  (1: non contact, i.e.out of max distance www.pnas.org/cgi/doi/10.1073/pnas.1914677117)
    because phi and theta is asymmetry, both two for loop begins from zero
    """
    L = mat.shape[0]
    bins = list(np.arange(-180, 180 + 15, 15))
    Nbin = (len(bins) - 1) + 1  # 24+1
    result = np.zeros([L, L, Nbin])
    result[..., 0] = (mat >= bins[0]).astype(float) * (mat <= bins[1]).astype(float)
    for i in range(1, Nbin - 1):
        result[..., i] = (mat > bins[i]).astype(float) * (mat <= bins[i + 1]).astype(float)

    contact = (distmat <= dmax).astype(float)  # 此时nan被mask掉，设置为0
    contact = np.tile(contact[..., None], [1, 1, Nbin - 1])
    result[..., : Nbin - 1] = result[..., : Nbin - 1] * contact
    noncontact = (distmat > dmax).astype(float)  # 此时nan被mask掉，设置为0
    masknan = (mat >= -180).astype(float)  # mask nan of omega matrix
    result[..., -1] = noncontact * masknan
    return result


def theta2bins(mat, distmat, dmax=30):
    return omega2bins(mat, distmat, dmax=30)


def phi2bins(mat, distmat, dmax=30):
    """
    This function is for label
    bins definition: left open, right close
    anglebin=(beg0,end0]
    first bin=[0,15]
    Nbin=12+1  (+non contact, i.e.out of max distance www.pnas.org/cgi/doi/10.1073/pnas.1914677117)
    because phi and theta is asymmetry, both two for loop begins from zero
    """
    L = mat.shape[0]
    bins = list(np.arange(0, 180 + 15, 15))
    Nbin = (len(bins) - 1) + 1
    result = np.zeros([L, L, Nbin])
    result[..., 0] = (mat >= bins[0]).astype(float) * (mat <= bins[1]).astype(float)
    for i in range(1, Nbin - 1):
        result[..., i] = (mat > bins[i]).astype(float) * (mat <= bins[i + 1]).astype(float)
    contact = (distmat <= dmax).astype(float)
    contact = np.tile(contact[..., None], [1, 1, Nbin - 1])
    result[..., : Nbin - 1] = result[..., : Nbin - 1] * contact
    noncontact = (distmat > dmax).astype(float)
    masknan = (mat >= 0).astype(float)
    result[..., -1] = noncontact * masknan
    return result


def cbLabelIndex(mat):
    cbBins = dist2bins(mat)
    L = mat.shape[0]
    binIndex = np.tile(np.arange(48)[None, None, :], [L, L, 1])
    cbIndex = np.array(np.sum(cbBins * binIndex, axis=2), dtype=int)
    return cbIndex


def omegaLabelIndex(mat, distmat):
    omegaBins = omega2bins(mat, distmat)
    L = mat.shape[0]
    binIndex = np.tile(np.arange(25)[None, None, :], [L, L, 1])
    omegaIndex = np.array(np.sum(omegaBins * binIndex, axis=2), dtype=int)
    return omegaIndex


def thetaLabelIndex(mat, distmat):
    return omegaLabelIndex(mat, distmat)


def phiLabelIndex(mat, distmat):
    phiBins = phi2bins(mat, distmat)
    L = mat.shape[0]
    binIndex = np.tile(np.arange(13)[None, None, :], [L, L, 1])
    phiIndex = np.array(np.sum(phiBins * binIndex, axis=2), dtype=int)
    return phiIndex


def get_dihedrals(a, b, c, d):
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:, None]

    v = b0 - np.sum(b0 * b1, axis=-1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=-1)[:, None] * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)

    return np.arctan2(y, x)


def get_angles(a, b, c):
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:, None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:, None]

    x = np.sum(v * w, axis=1)

    return np.arccos(x)


def get_info(xyz, dmax):
    """This function is adapted from codes of Jianyi Yang et al. PNAS, 2020.(https://doi.org/10.1073/pnas.1914677117)"""

    nres = xyz.shape[0]
    N = xyz[:, 0, :]
    CA = xyz[:, 1, :]
    C = xyz[:, 2, :]
    CB = xyz[:, 3, :]

    d_CA_CB = CB - CA
    d_CA_CB = np.sqrt(np.sum(d_CA_CB * d_CA_CB, axis=-1))
    gly_index = np.where(d_CA_CB < 0.001)[0]

    # fast neighbors search
    kdCb = scipy.spatial.cKDTree(CB)
    indices = kdCb.query_ball_tree(kdCb, dmax)
    # indices of contacting residues
    idx = np.array([[i, j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]

    # Cb-Cb distance matrix
    dist6d_CB = np.zeros((nres, nres))
    dist6d_CB[idx0, idx1] = np.linalg.norm(CB[idx1] - CB[idx0], axis=-1)

    # Ca-Ca distance matrix
    dist6d_CA = np.zeros((nres, nres))
    dist6d_CA[idx0, idx1] = np.linalg.norm(CA[idx1] - CA[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.full([nres, nres], np.nan)
    omega6d[idx0, idx1] = np.rad2deg(get_dihedrals(CA[idx0], CB[idx0], CB[idx1], CA[idx1]))
    omega6d[gly_index, :] = np.nan
    omega6d[:, gly_index] = np.nan

    # matrix of polar coord theta
    theta6d = np.full([nres, nres], np.nan)
    theta6d[idx0, idx1] = np.rad2deg(get_dihedrals(N[idx0], CA[idx0], CB[idx0], CB[idx1]))
    theta6d[gly_index, :] = np.nan
    theta6d[:, gly_index] = np.nan

    # matrix of polar coord phi
    phi6d = np.full([nres, nres], np.nan)
    phi6d[idx0, idx1] = np.rad2deg(get_angles(CA[idx0], CB[idx0], CB[idx1]))
    phi6d[gly_index, :] = np.nan
    phi6d[:, gly_index] = np.nan

    return (dist6d_CB, dist6d_CA, omega6d, theta6d, phi6d)


def NormVec(V):
    eps = 1e-10
    axis_x = V[:, 2] - V[:, 1]
    axis_x /= torch.norm(axis_x, dim=-1).unsqueeze(1) + eps
    axis_y = V[:, 0] - V[:, 1]
    axis_z = torch.cross(axis_x, axis_y, dim=1)
    axis_z /= torch.norm(axis_z, dim=-1).unsqueeze(1) + eps
    axis_y = torch.cross(axis_z, axis_x, dim=1)
    Vec = torch.stack([axis_x, axis_y, axis_z], dim=1)
    return Vec


def comp_feature(atoms):
    # atom: torch.tensor()  shape:[L, 4, 3] -> [L, [N, CA, C, CB], [x, y, z]]
    atoms_numpy = atoms
    atoms = torch.tensor(atoms)
    N_CA_C = atoms[:, :-1, :]  # [L, [N, CA, C, CB], [x, y, z]]
    rotation = NormVec(N_CA_C)  # N_CA_C, atoms=(L,3,3)
    r = torch.linalg.inv(rotation)

    xyz_CA = torch.einsum("a b i, a i j -> a b j", atoms[:, 1].unsqueeze(0) - atoms[:, 1].unsqueeze(1), r)  # (L,L,3)

    CB_dist, CA_dist, omega, theta, phi = get_info(atoms_numpy, dmax=10000)
    CA_dist = CA_dist[None]  # (1,L,L)
    cbIndex = cbLabelIndex(CB_dist)[None]
    omegaIndex = omegaLabelIndex(omega, CB_dist)[None]
    thetaIndex = thetaLabelIndex(theta, CB_dist)[None]
    phiIndex = phiLabelIndex(phi, CB_dist)[None]
    label_5 = np.concatenate([CA_dist, cbIndex, omegaIndex, thetaIndex, phiIndex], axis=0)
    label_5 = torch.from_numpy(label_5)

    label_all = torch.cat([label_5, xyz_CA.permute(2, 0, 1)], dim=0)
    return label_all  # (8,L,L)


# transfer sequence from string to digital tokens
def seq2tokens(seq_list, amino_acid_dict):
    seq_tokens_list = []
    for sequence in seq_list:
        token_list = []
        for r in sequence:
            if r in amino_acid_dict.keys():
                token_list.append(amino_acid_dict[r])
            else:
                token_list.append(20)  # unknown/rare amino acid
        seq_tokens_list.append(token_list)
    seq_tokens_array = np.array(seq_tokens_list)
    return seq_tokens_array


DICT = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "N",
    "GLX": "Q",
    "UNK": "G",
    "HSD": "H",
}


def parse_pdb(pdbfile, chain_number=1):
    dict_pdb = defaultdict(list)
    fr = open(pdbfile)
    lines = fr.readlines()
    for line in lines:
        if line.startswith("ATOM"):
            chain = line[21]  # example: chain A
            dict_pdb[chain].append(line)
    chains = list(dict_pdb.keys())
    key = chains[chain_number - 1]
    return dict_pdb[key]


def get_xyz(pdb):
    target = ["N", "CA", "C", "CB"]
    dict_xyz_all = defaultdict(dict)
    dict_xyz = defaultdict(dict)
    calculated_sequence = []

    for line in pdb:
        residue = line[17:20].strip()  # residue ASN
        ####remove nonstandard amino acid
        if DICT.get(residue):
            atomname = line[12:16].strip()  # name of the atom
            residue_abbrv = DICT[residue]  # abbreviation of the residue
            if atomname in target:
                res_no = line[22:26].strip()  # residue number
                coords_str = [line[30:38], line[38:46], line[46:54]]
                coords = [float(k) for k in coords_str]

                if res_no not in dict_xyz_all.keys():
                    calculated_sequence.append(residue_abbrv)

                dict_xyz_all[res_no][atomname] = coords  # dict_xyz_all[residual_number_ID][atom_name] = coords

    coords_list = []

    for res in dict_xyz_all.keys():
        for atom in ["N", "CA", "C", "CB"]:
            if atom not in dict_xyz_all[res].keys():
                dict_xyz_all[res][atom] = [np.nan, np.nan, np.nan]
        res_coords = np.array([dict_xyz_all[res][atom] for atom in ["N", "CA", "C", "CB"]])
        coords_list.append(res_coords)
    coords = np.stack(coords_list)

    coords = np.nan_to_num(coords, copy=True, nan=-99999)

    N = coords[:, 0, :]  # N=(L, 3)
    CA = coords[:, 1, :]
    C = coords[:, 2, :]
    CB = coords[:, 3, :]

    CB_nanIndex = np.where(CB < -9999)

    CB[CB_nanIndex] = CA[CB_nanIndex]

    coords[:, 3, :] = CB

    return coords, "".join(calculated_sequence)


if __name__ == "__main__":

    sample_list_file = opt.sample_list_file
    pred_pdb_dir = opt.pred_pdb_dir
    # fasta_dir = opt.fasta_dir
    true_coords_hdf5_path = opt.true_coords_hdf5

    # read sample list
    sample_list = []
    with open(sample_list_file, "r") as file:
        for line in file.readlines():
            protein_name = line.strip().split("\t")[-1]
            sample_list.append(protein_name)

    # create hdf5 file
    true_coords_hdf5 = h5py.File(true_coords_hdf5_path, "w")

    length_stat = open("length_stat.txt", "w")

    lDDT_dict = {}

    for domain in sample_list:

        subclass = domain[1:3]

        print("\n" + domain)

        pred_pdb_path = os.path.join(pred_pdb_dir, "{}.pdb".format(domain))
        if not os.path.exists(pred_pdb_path):
            print("pdb file not found")
            continue

        coords, sequence = get_xyz(parse_pdb(pred_pdb_path))  # coords = (L, 3, 3)
        print(sequence)
        target_tokens = seq2tokens([sequence], seq2token_dict)
        print("coords", coords.shape, "target_tokens", target_tokens.shape)

        # remove nan
        non_nan = np.all(np.all(coords > -9999, axis=-1), axis=-1)  # (L,)
        coords = coords[non_nan, :, :]  # (L,4,3)

        # compute label
        labelall = comp_feature(coords)  # (8, L, L)
        len_pdb_seq = labelall.shape[-1]

        # target_tokens
        target_tokens = target_tokens[:, non_nan]

        # calculate phi psi
        label_Psi_Phi = PsiPhi(torch.tensor(coords))
        label_Psi_Phi = label_Psi_Phi.cpu().detach().numpy()
        # print("label_Psi_Phi", label_Psi_Phi.shape)

        length_stat.write(f"{domain}\t{len_pdb_seq}\n")

        subgroup = true_coords_hdf5.create_group(domain)
        subgroup.create_dataset("xyz", data=coords, dtype=np.float32)
        # subgroup.create_dataset('labelall', data=labelall, dtype=np.float32)
        subgroup.create_dataset("target_tokens", data=target_tokens, dtype=np.int8)
        subgroup.create_dataset("Psi_Phi", data=label_Psi_Phi, dtype=np.float32)

    true_coords_hdf5.close()
    length_stat.close()
