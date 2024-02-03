import os
import click
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from scripts.model import SPIRED_Stab
from scripts.utils_train_valid import getStabDataTest

aa_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', 'X':'ALA'}

working_directory = os.path.abspath(os.path.dirname(__file__))

@click.command()
@click.option('--fasta_file', required = True, type = str)
@click.option('--saved_folder', required = True, type = str)
def main(fasta_file, saved_folder):
    
    # load parameter
    model = SPIRED_Stab(device_list = ['cpu', 'cpu', 'cpu', 'cpu'])
    model.load_state_dict(torch.load(f'{working_directory}/model/SPIRED-Stab.pth'))
    model.eval()
    
    # load ESM-2 650M model
    esm2_650M, _ = torch.hub.load('facebookresearch/esm:main', 'esm2_t33_650M_UR50D')
    esm2_650M.eval()
    
    # load ESM-2 3B model
    esm2_3B, esm2_alphabet = torch.hub.load('facebookresearch/esm:main', 'esm2_t36_3B_UR50D')
    esm2_3B.eval()
    esm2_batch_converter = esm2_alphabet.get_batch_converter()
    
    # save sequence information
    wt_seq = str(list(SeqIO.parse(fasta_file, 'fasta'))[0].seq)
    mut_seq = str(list(SeqIO.parse(fasta_file, 'fasta'))[1].seq)
    
    mut_pos_torch_list = torch.tensor((np.array(list(wt_seq)) != np.array(list(mut_seq))).astype(int).tolist())
    
    # predict
    with torch.no_grad():
        os.makedirs(f'{saved_folder}/wt/CA_structure', exist_ok = True)
        os.makedirs(f'{saved_folder}/wt/GDFold2', exist_ok = True)
        os.makedirs(f'{saved_folder}/mut/CA_structure', exist_ok = True)
        os.makedirs(f'{saved_folder}/mut/GDFold2', exist_ok = True)
        
        # data
        f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTest(wt_seq, esm2_3B, esm2_650M, esm2_batch_converter)
        wt_data = {
            'target_tokens': target_tokens,
            'esm2-3B': f1d_esm2_3B,
            'embedding': f1d_esm2_650M
        }
        f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTest(mut_seq, esm2_3B, esm2_650M, esm2_batch_converter)
        mut_data = {
            'target_tokens': target_tokens,
            'esm2-3B': f1d_esm2_3B,
            'embedding': f1d_esm2_650M
        }
        ddG, dTm, wt_features, mut_features = model(wt_data, mut_data, mut_pos_torch_list)
        
        # write pred value
        pd.DataFrame({'ddG': ddG.item(), 'dTm': dTm.item()}, index = [0]).to_csv(f'{saved_folder}/pred.csv', index = False)
        
        # write wt npz for GDFold2
        Predxyz_4thBlock_6thlayer = wt_features['Predxyz']['4th'][-1]
        plddt_value = wt_features['Plddt']['4th'][-1][0]
        plddt_value_L = torch.mean(plddt_value, dim = 1)
        plddt_top8_idx = torch.topk(plddt_value_L, 8)[-1]
        plddt_value_top8 = plddt_value[plddt_top8_idx, :]
        xyz_top8 = Predxyz_4thBlock_6thlayer[0, :, plddt_top8_idx, :]
        xyz_top8 = xyz_top8.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
        phi_psi_1D = wt_features['phi_psi_1D'][0].permute(1,0).cpu().detach().numpy().astype(np.float32)
        plddt_top8_idx = plddt_top8_idx.cpu().detach().numpy().astype(np.int32)
        plddt_value_top8 = plddt_value_top8.cpu().detach().numpy().astype(np.float32)
        np.savez(f'{saved_folder}/wt/GDFold2/input.npz', reference = plddt_top8_idx, translation = xyz_top8, dihedrals = phi_psi_1D, plddt = plddt_value_top8)
        
        # write wt pdb
        N, L, _ = xyz_top8.shape
        if N > 8:
            N = 8
        
        for n in range(N):
            xyz_L = xyz_top8[n, ...]
            with open(f'{saved_folder}/wt/CA_structure/{n}.pdb', 'w') as f:
                for i in range(L):
                    amino_acid = aa_dict[wt_seq[i]]
                    xyz_ca = xyz_L[i, ...]
                    x, y, z = (round(float(xyz_ca[0]), 3), round(float(xyz_ca[1]),3), round(float(xyz_ca[2]), 3))
                    f.write('ATOM  {:>5} {:<4} {} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2}  \n'.format(int(i + 1), 'CA', amino_acid, int(i + 1), x, y, z, 1.0, 0.0, 'C'))
        
        # write mut npz for GDFold2
        Predxyz_4thBlock_6thlayer = mut_features['Predxyz']['4th'][-1]
        plddt_value = mut_features['Plddt']['4th'][-1][0]
        plddt_value_L = torch.mean(plddt_value, dim = 1)
        plddt_top8_idx = torch.topk(plddt_value_L, 8)[-1]
        plddt_value_top8 = plddt_value[plddt_top8_idx, :]
        xyz_top8 = Predxyz_4thBlock_6thlayer[0, :, plddt_top8_idx, :]
        xyz_top8 = xyz_top8.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
        phi_psi_1D = mut_features['phi_psi_1D'][0].permute(1,0).cpu().detach().numpy().astype(np.float32)
        plddt_top8_idx = plddt_top8_idx.cpu().detach().numpy().astype(np.int32)
        plddt_value_top8 = plddt_value_top8.cpu().detach().numpy().astype(np.float32)
        np.savez(f'{saved_folder}/mut/GDFold2/input.npz', reference = plddt_top8_idx, translation = xyz_top8, dihedrals = phi_psi_1D, plddt = plddt_value_top8)
        
        # write mut pdb
        N, L, _ = xyz_top8.shape
        if N > 8:
            N = 8
        
        for n in range(N):
            xyz_L = xyz_top8[n, ...]
            with open(f'{saved_folder}/mut/CA_structure/{n}.pdb', 'w') as f:
                for i in range(L):
                    amino_acid = aa_dict[mut_seq[i]]
                    xyz_ca = xyz_L[i, ...]
                    x, y, z = (round(float(xyz_ca[0]), 3), round(float(xyz_ca[1]),3), round(float(xyz_ca[2]), 3))
                    f.write('ATOM  {:>5} {:<4} {} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2}  \n'.format(int(i + 1), 'CA', amino_acid, int(i + 1), x, y, z, 1.0, 0.0, 'C'))

if __name__ == '__main__':
    main()
