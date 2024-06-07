import os
import click
import torch
import itertools
import numpy as np
import pandas as pd
from Bio import SeqIO
from scripts.model import SPIRED_Fitness_Union
from scripts.utils_train_valid import getDataTest

amino_acid_list = list("ARNDCQEGHILKMFPSTWYV")
amino_acid_dict = {}
for index, value in enumerate(amino_acid_list):
    amino_acid_dict[index] = value

double_mut_list = list(itertools.product(amino_acid_list, amino_acid_list, repeat=1))
double_mut_dict = {}
double_mut_dict_inverse = {}
for index, value in enumerate(double_mut_list):
    double_mut_dict[index] = "".join(value)
    double_mut_dict_inverse["".join(value)] = index

aa_dict = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL", "X": "ALA"}

working_directory = os.path.abspath(os.path.dirname(__file__))


@click.command()
@click.option("--fasta_file", required=True, type=str)
@click.option("--saved_folder", required=True, type=str)
def main(fasta_file, saved_folder):

    # load parameter
    model = SPIRED_Fitness_Union(device_list=["cpu", "cpu", "cpu", "cpu"])
    model.load_state_dict(torch.load(f"{working_directory}/model/SPIRED-Fitness.pth"))
    model.eval()

    # load ESM-2 650M model
    esm2_650M, _ = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    esm2_650M.eval()

    # load ESM-2 3B model
    esm2_3B, esm2_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t36_3B_UR50D")
    esm2_3B.eval()
    esm2_batch_converter = esm2_alphabet.get_batch_converter()

    # load 5 ESM-1v models
    esm1v_1, _ = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_1")
    esm1v_2, _ = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_2")
    esm1v_3, _ = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_3")
    esm1v_4, _ = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_4")
    esm1v_5, esm1v_alphabet = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_5")
    esm1v_1.eval()
    esm1v_2.eval()
    esm1v_3.eval()
    esm1v_4.eval()
    esm1v_5.eval()
    esm1v_batch_converter = esm1v_alphabet.get_batch_converter()

    # save multiple sequence information
    fasta_dict = {}
    for i in SeqIO.parse(fasta_file, "fasta"):
        fasta_dict[i.description] = str(i.seq)

    # predict
    with torch.no_grad():
        for protein_name in fasta_dict.keys():
            os.makedirs(f"{saved_folder}/{protein_name}/CA_structure", exist_ok=True)
            os.makedirs(f"{saved_folder}/{protein_name}/GDFold2", exist_ok=True)
            os.makedirs(f"{saved_folder}/{protein_name}/features_for_downstream", exist_ok=True)

            seq = fasta_dict[protein_name]

            f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, target_tokens = getDataTest(seq, esm2_3B, esm2_650M, esm1v_1, esm1v_2, esm1v_3, esm1v_4, esm1v_5, esm1v_batch_converter, esm1v_alphabet, esm2_batch_converter)

            single_pred, double_pred, Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, f1d_cycle, f2d_cycle = model(target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits)

            # write 3d and plddt pt
            torch.save(Predxyz["4th"][-1][0].permute(1, 2, 0).detach().cpu().clone(), f"{saved_folder}/{protein_name}/features_for_downstream/3d.pt")
            torch.save(Plddt["4th"][-1][0].detach().cpu().clone(), f"{saved_folder}/{protein_name}/features_for_downstream/plddt.pt")
            torch.save(double_pred[0].detach().cpu().clone(), f"{saved_folder}/{protein_name}/double_mut_pred.pt")

            # write pred csv
            pred = single_pred[0].detach().cpu().clone()
            data = pd.DataFrame(columns=amino_acid_list)
            for i in range(len(seq)):
                for j in range(len(amino_acid_list)):
                    data.loc[i, amino_acid_list[j]] = pred[i, j].item()
            data.index = list(seq)
            data.to_csv(f"{saved_folder}/{protein_name}/single_mut_pred_for_heatmap.csv")

            data = pd.DataFrame(columns=["pred_score"])
            for mut_pos, wt_res in enumerate(seq):
                for mut_res in amino_acid_list:
                    mut_info = f"{wt_res}{mut_pos}{mut_res}"
                    if wt_res != mut_res:
                        data.loc[mut_info, "pred_score"] = pred[mut_pos, amino_acid_list.index(mut_res)].item()
            data.to_csv(f"{saved_folder}/{protein_name}/single_mut_pred.csv")

            pred = double_pred[0].detach().cpu().clone()
            upper_bound = torch.topk(pred.flatten(), k=2000, largest=True)[0][-1]
            lower_bound = torch.topk(pred.flatten(), k=2000, largest=False)[0][-1]

            # top k
            data = pd.DataFrame(columns=["pred_score"])
            x, y, z = torch.where(pred >= upper_bound)
            x, y, z = x.tolist(), y.tolist(), z.tolist()
            for index in range(len(x)):
                mut_info = seq[x[index]] + str(x[index]) + double_mut_dict[z[index]][0] + "," + seq[y[index]] + str(y[index]) + double_mut_dict[z[index]][-1]
                if x[index] != y[index] and seq[x[index]] != double_mut_dict[z[index]][0] and seq[y[index]] != double_mut_dict[z[index]][-1]:
                    mut_info_inverse = seq[y[index]] + str(y[index]) + double_mut_dict[z[index]][-1] + "," + seq[x[index]] + str(x[index]) + double_mut_dict[z[index]][0]
                    score = pred[x[index], y[index], z[index]].item()
                    score += pred[y[index], x[index], double_mut_dict_inverse[double_mut_dict[z[index]][-1] + double_mut_dict[z[index]][0]]].item()
                    score /= 2
                    if mut_info not in data.index and mut_info_inverse not in data.index:
                        data.loc[mut_info, "pred_score"] = score
            data.to_csv(f"{saved_folder}/{protein_name}/double_mut_pred_top_k.csv")

            # down k
            data = pd.DataFrame(columns=["pred_score"])
            x, y, z = torch.where(pred <= lower_bound)
            x, y, z = x.tolist(), y.tolist(), z.tolist()
            for index in range(len(x)):
                mut_info = seq[x[index]] + str(x[index]) + double_mut_dict[z[index]][0] + "," + seq[y[index]] + str(y[index]) + double_mut_dict[z[index]][-1]
                if x[index] != y[index] and seq[x[index]] != double_mut_dict[z[index]][0] and seq[y[index]] != double_mut_dict[z[index]][-1]:
                    mut_info_inverse = seq[y[index]] + str(y[index]) + double_mut_dict[z[index]][-1] + "," + seq[x[index]] + str(x[index]) + double_mut_dict[z[index]][0]
                    score = pred[x[index], y[index], z[index]].item()
                    score += pred[y[index], x[index], double_mut_dict_inverse[double_mut_dict[z[index]][-1] + double_mut_dict[z[index]][0]]].item()
                    score /= 2
                    if mut_info not in data.index and mut_info_inverse not in data.index:
                        data.loc[mut_info, "pred_score"] = score
            data.to_csv(f"{saved_folder}/{protein_name}/double_mut_pred_last_k.csv")

            # write npz for GDFold2
            Predxyz_4thBlock_6thlayer = Predxyz["4th"][-1]
            plddt_value = Plddt["4th"][-1][0]
            plddt_value_L = torch.mean(plddt_value, dim=1)
            plddt_top8_idx = torch.topk(plddt_value_L, 8)[-1]
            plddt_value_top8 = plddt_value[plddt_top8_idx, :]
            xyz_top8 = Predxyz_4thBlock_6thlayer[0, :, plddt_top8_idx, :]
            xyz_top8 = xyz_top8.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
            phi_psi_1D = phi_psi_1D[0].permute(1, 0).cpu().detach().numpy().astype(np.float32)
            plddt_top8_idx = plddt_top8_idx.cpu().detach().numpy().astype(np.int32)
            plddt_value_top8 = plddt_value_top8.cpu().detach().numpy().astype(np.float32)
            np.savez(f"{saved_folder}/{protein_name}/GDFold2/input.npz", reference=plddt_top8_idx, translation=xyz_top8, dihedrals=phi_psi_1D, plddt=plddt_value_top8)

            # write pdb
            N, L, _ = xyz_top8.shape
            if N > 8:
                N = 8

            for n in range(N):
                xyz_L = xyz_top8[n, ...]
                with open(f"{saved_folder}/{protein_name}/CA_structure/{n}.pdb", "w") as f:
                    for i in range(L):
                        amino_acid = aa_dict[seq[i]]
                        xyz_ca = xyz_L[i, ...]
                        x, y, z = (round(float(xyz_ca[0]), 3), round(float(xyz_ca[1]), 3), round(float(xyz_ca[2]), 3))
                        f.write("ATOM  {:>5} {:<4} {} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2}  \n".format(int(i + 1), "CA", amino_acid, int(i + 1), x, y, z, 1.0, 0.0, "C"))


if __name__ == "__main__":
    main()
