import torch
import esm
import numpy as np
import h5py, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--PDB_coords_seq_hdf5", type=str, default="./", help="path for train hdf5 file for coordinates")
parser.add_argument("--gpu_esm", type=int, default=[3], nargs="+", help="the IDs of GPU to be used for ESM2 model")
parser.add_argument("--train_list", type=str, default="./", help="path for train sample list file")
parser.add_argument("--ESM_length_cutoff", type=int, default=500, help="Max lehgth of protein sequence for ESM2 model")
parser.add_argument("--out_hdf5", type=str, default="./", help="path for train hdf5 file for coordinates")
FLAGS = parser.parse_args()

PDB_coords_seq_hdf5 = h5py.File(FLAGS.PDB_coords_seq_hdf5, "r")
out_hdf5 = h5py.File(FLAGS.out_hdf5, "w")


def tokens2seq(token_array):
    # X: rare amino acid or unknown amino acid;  "-": gap;
    amino_acid_dict = {
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
    msa_sequence_list = []
    row, col = token_array.shape
    for i in range(row):
        token_list = []
        for k in range(col):
            token = amino_acid_dict[token_array[i, k]]
            token_list.append(token)
        sequence = "".join(token_list)
        msa_sequence_list.append((str(i), sequence))
    return msa_sequence_list


def ESM2_embed(seq_tokens, model_CPU, model_GPU, length_cutoff, gpu):
    # pred_seq = (batch, L) differentialbel one-hot
    batch, L = seq_tokens.shape
    L = L - 2
    with torch.no_grad():
        if L > length_cutoff:
            print("L", L, "length_cutoff", length_cutoff, "CPU")
            results = model_CPU(seq_tokens.to(device="cpu"), repr_layers=range(37), need_head_weights=False, return_contacts=True)
        else:
            print("L", L, "length_cutoff", length_cutoff, "GPU")
            results = model_GPU(seq_tokens.to(device=gpu), repr_layers=range(37), need_head_weights=False, return_contacts=True)

    token_embeds = torch.stack([v for _, v in sorted(results["representations"].items())], dim=2)
    token_embeds = token_embeds[:, 1:-1]  # (batch, L, 37, dim=2560)
    token_embeds = token_embeds.to(device=gpu, dtype=torch.float32)  # (batch, L, dim=2560)
    # print("token_embeds", token_embeds.shape)

    ###### attention map and contact map ######
    attentions = results["attentions"]  # (batch, layers=36, heads=40, L+2, L+2)
    attentions = attentions[:, -1, :, 1:-1, 1:-1]  # (batch, 40, L, L)
    contacts = results["contacts"].unsqueeze(1)  # (batch, 1, L, L)
    return (token_embeds, attentions)


# ESM2-3B
device_list_esm = ["cuda:" + str(gpu) for gpu in FLAGS.gpu_esm]
esm_model_path = "/export/disk4/chenyinghui/database/Evolutionary_Scale_Modeling/esm2_t36_3B_UR50D.pt"
esm2_CPU, alphabet_esm2 = esm.pretrained.load_model_and_alphabet(esm_model_path)
esm2_GPU, alphabet_esm2 = esm.pretrained.load_model_and_alphabet(esm_model_path)
batch_converter_esm2 = alphabet_esm2.get_batch_converter()
device_esm = torch.device(device_list_esm[0])
esm2_GPU = esm2_GPU.to(device=device_esm)
esm2_CPU.eval()
esm2_GPU.eval()

# read sample list
train_list = []
with open(FLAGS.train_list, "r") as FILE:
    for line in FILE:
        train_list.append(line.strip().split("\t")[0])

for domain in train_list:
    target_tokens = PDB_coords_seq_hdf5[domain]["target_tokens"][:]  # (1,L)
    seq_str = tokens2seq(target_tokens)
    labels, strs, target_tokens = batch_converter_esm2(seq_str)  # target_tokens = (batch, L+2 )
    f1d_esm2_3B, f2dbatch = ESM2_embed(target_tokens, esm2_CPU, esm2_GPU, length_cutoff=FLAGS.ESM_length_cutoff, gpu=device_esm)  # f1dbatch=(N,L,C), f2dbatch=(N,C,L,L)
    out_hdf5.create_group(domain)
    out_hdf5[domain].create_dataset("f1d_esm2_3B", data=f1d_esm2_3B.cpu().numpy(), dtype=np.float32)

PDB_coords_seq_hdf5.close()
out_hdf5.close()
