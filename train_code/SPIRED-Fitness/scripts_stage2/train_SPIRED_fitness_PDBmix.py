# -*- coding: utf-8 -*-
import sys, os

env_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(env_path + "/../")
sys.path.append(env_path + "/seq2struct")
sys.path.append(env_path + "/seq2struct/esmfold_openfold")

import torch
import numpy as np
import pandas as pd

import re, time, random, argparse, h5py
import esm
from seq2struct.model import SPIRED_Fitness_Union
from seq2struct.utils_train_valid import threadTrain, threadValid, getData_fitness, getData_PDB, train_loss_dict, valid_loss_dict, make_loss_dict, write_pdb, makelog, write_loss_table


parser = argparse.ArgumentParser()
parser.add_argument("--PDB_coords_seq_hdf5", type=str, default="./", help="hdf5 file path for PDB data of train set")
parser.add_argument("--CATH_coords_seq_hdf5", type=str, default="./", help="hdf5 file path for CATH data of train set")
parser.add_argument("--fit_coords_seq_hdf5", type=str, default="./", help="hdf5 file for coordinates of fitness samples")
parser.add_argument("--valid_coords_seq_hdf5", type=str, default="./", help="hdf5 file path for coords and seqs data of validation set")
parser.add_argument("--esm2_3B_hdf5", type=str, default="./", help="train hdf5 file for ESM2 3B feature")
parser.add_argument("--fitness_data_h5", type=str, default="./", help="pt file for fitness label and embedding from ESM2-650M and ESM-1v.")
parser.add_argument("--train_list", type=str, default="./", help="path for train sample list file")
parser.add_argument("--valid_list", type=str, default="./", help="path for validation sample list file")
parser.add_argument("--fitness_list", type=str, default="./", help="path for fitness sample list file")
parser.add_argument("--train_length", type=int, default=256, help="Max length of protein sequence for model training")
parser.add_argument("--valid_length", type=int, default=800, help="Max length of protein sequence for validation")
parser.add_argument("--train_cycle", type=int, default=1, help="cycle number for train")
parser.add_argument("--valid_cycle", type=int, default=1, help="cycle number for validation")
parser.add_argument("--batch", type=int, default=1, help="batch size for training SPIRED-Fitness, default=1")
parser.add_argument("--ESM_length_cutoff", type=int, default=400, help="Max length of protein sequence for ESM2 model inference")
parser.add_argument("--epoch_start", type=int, default=1, help="number of start training epochs")
parser.add_argument("--epoch_end", type=int, default=30, help="number of end training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--gpu_spired", type=int, default=[0, 1], nargs="+", help="the IDs of GPU to be used for SPIRED")
parser.add_argument("--gpu_esm", type=int, default=[3], nargs="+", help="the IDs of GPU to be used for ESM2 model")
parser.add_argument("--cpu", type=int, default=0, help="use cpu to train (1) or not (0)")
parser.add_argument("--SPIRED_checkpoint", type=str, default="None", help="path of saved parameters of SPIRED")
parser.add_argument("--Fitness_checkpoint", type=str, default="None", help="path of saved parameters of fitness model")
parser.add_argument("--unionmodel_checkpoint", type=str, default="None", help="path of saved parameters of SPIRED-Fitness model")
parser.add_argument("--train_sample_num", type=int, default=3000, help="sample number for training set")
parser.add_argument("--valid_sample_num", type=int, default=300, help="sample number for validation set")
parser.add_argument("--weight_loss_struct", type=float, default=0.01, help="weight for structure loss")
parser.add_argument("--weight_loss_spearman", type=float, default=1, help="weight for structure loss")
parser.add_argument("--model_dir", type=str, default="./", help="directory for saving models")
parser.add_argument("--losslog", type=str, default="./struct_loss_log.xls", help="path for struct loss log")
parser.add_argument("--write_pdb", type=int, default=1, help="write pdb file (1) or not(0)")
FLAGS = parser.parse_args()


## load data
PDB_coords_seq_hdf5 = h5py.File(FLAGS.PDB_coords_seq_hdf5, "r")
CATH_coords_seq_hdf5 = h5py.File(FLAGS.CATH_coords_seq_hdf5, "r")
fit_coords_seq_hdf5 = h5py.File(FLAGS.fit_coords_seq_hdf5, "r")
valid_coords_seq_hdf5 = h5py.File(FLAGS.valid_coords_seq_hdf5, "r")
esm2_3B_hdf5 = h5py.File(FLAGS.esm2_3B_hdf5, "r")
fitness_data_h5 = h5py.File(FLAGS.fitness_data_h5)

valid_set, fitness_sample_list = ([], [])

with open(FLAGS.valid_list, "r") as valid_list:
    for line in valid_list:
        sample_name = re.split("\t", line.strip())[0]
        valid_set.append(sample_name)

with open(FLAGS.fitness_list, "r") as LIST:
    for line in LIST:
        sample_name = re.split("\t", line.strip())[0]
        fitness_sample_list.append(sample_name)

valid_set = valid_set[: FLAGS.valid_sample_num]


## GPU device
if FLAGS.cpu == 0:
    device_list_spired = ["cuda:" + str(gpu) for gpu in FLAGS.gpu_spired]
    device_list_esm = ["cuda:" + str(gpu) for gpu in FLAGS.gpu_esm]
else:
    device_list_spired = ["cpu"]
    device_list_esm = ["cpu"]


device1_spired = torch.device(device_list_spired[0])
device2_spired = torch.device(device_list_spired[-1])
device_esm = torch.device(device_list_esm[0])

# token to sequence
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

# random parameter initialization
union_model = SPIRED_Fitness_Union(device_list=device_list_spired)

# get union_model parameters
union_model_state_dict = union_model.state_dict()

if FLAGS.unionmodel_checkpoint != "None":
    union_model_checkpoint = torch.load(FLAGS.unionmodel_checkpoint, map_location="cpu")
    union_model_state_dict = union_model_checkpoint["net"].copy()
    optimizer_state_dict = union_model_checkpoint["optimizer"].copy()
else:
    # SPIRED model
    SPIRED_model_state_dict = torch.load(FLAGS.SPIRED_checkpoint, map_location="cpu").copy()

    # add "SPIRED" prefix to the parameters of SPIRED model
    SPIRED_model_state_dict = {("SPIRED." + k): v for k, v in SPIRED_model_state_dict.items() if ("SPIRED." + k in union_model_state_dict and v.shape == union_model_state_dict["SPIRED." + k].shape)}
    union_model_state_dict.update(SPIRED_model_state_dict)

    if FLAGS.Fitness_checkpoint != "None":
        # Fitness model
        Fitness_model_state_dict = torch.load(FLAGS.Fitness_checkpoint, map_location="cpu").state_dict().copy()

        # add "Fitness" prefix to the parameters of Fitness model
        Fitness_model_state_dict = {("Fitness." + k): v for k, v in Fitness_model_state_dict.items() if ("Fitness." + k in union_model_state_dict and v.shape == union_model_state_dict["Fitness." + k].shape)}
        union_model_state_dict.update(Fitness_model_state_dict)

union_model.load_state_dict(union_model_state_dict)

# fixed pretrain parameters
# for name, param in union_model.named_parameters():
#     print(name, param.requires_grad)

## Load ESM-2 model
esm_model_path = "esm2/esm2_t36_3B_UR50D.pt"
esm2_3B_CPU, alphabet_esm2 = esm.pretrained.load_model_and_alphabet(esm_model_path)
esm2_3B_GPU, alphabet_esm2 = esm.pretrained.load_model_and_alphabet(esm_model_path)
batch_converter_esm2 = alphabet_esm2.get_batch_converter()
esm2_3B_GPU = esm2_3B_GPU.to(device=device_esm)
esm2_3B_CPU.eval()
esm2_3B_GPU.eval()

# calculate number of parameters of the model
params_spired = sum(p.numel() for p in list(union_model.parameters())) / 1e6
print("Parameters of SPIRED-Fitness Model: %.3fM" % (params_spired))

## set different learning rate for SPIRED and Fitness modules
optimizer = torch.optim.Adam(
    [
        {"params": union_model.SPIRED.parameters(), "lr": 0.00001},
        {"params": union_model.Fitness.parameters(), "lr": FLAGS.lr},
    ],
    lr=FLAGS.lr,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=10, verbose=True)


def create_loss_dict(sample_name=False):
    dict = {
        "epoch": [],
        "batch": [],
        "sample_name": [],
        "loss_sum": [],
        "loss_struct": [],
        "soft_spearman_loss": [],
        "all_spearman_corr": [],
        "single_spearman_corr": [],
        "double_spearman_corr": [],
        "RD_loss_sum": [],
        "loss_phi_psi_1D": [],
        "loss_CE": [],
        "truelddt_median": [],
        "truelddt_max": [],
        "fape40_2": [],
    }
    if sample_name == False:
        del dict["batch"]
        del dict["sample_name"]
    else:
        del dict["epoch"]
    return dict


Loss_train_all_epochs, Loss_valid_all_epochs = (create_loss_dict(sample_name=False), create_loss_dict(sample_name=False))

# train and validation process
lr = FLAGS.lr
for epoch in range(FLAGS.epoch_start, FLAGS.epoch_end + 1):

    Loss_train_epoch, Loss_valid_epoch = (create_loss_dict(sample_name=True), create_loss_dict(sample_name=True))

    train_set = []
    with open(os.path.join(FLAGS.train_list, "epoch" + str(epoch) + ".txt"), "r") as train_list:
        for line in train_list:
            sample_name = re.split("\t", line.strip())[0]
            train_set.append(sample_name)

    # train_set = train_set[:FLAGS.train_sample_num]
    # shuffle train data set
    random.shuffle(train_set)

    ## set Generator into training mode
    union_model.train()

    sample_index = 0
    LossDict = make_loss_dict()

    ## training
    for i in range(0, len(train_set)):
        sample_index += 1
        print("sample_index", sample_index)

        domain = train_set[i]
        domain_batch = [domain]

        print("train_sample", domain)

        s0 = time.time()
        if domain in fitness_sample_list:
            single_label_bacth, double_label_bacth, single_index, double_index, label_8LL_batch, psi_phi_batch, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, target_tokens, target_seq, true_xyz = getData_fitness("train", domain, fit_coords_seq_hdf5, esm2_3B_hdf5, fitness_data_h5, batch_converter_esm2, device1_spired, device2_spired, FLAGS)
        else:
            if re.search("_", domain):
                single_label_bacth, double_label_bacth, single_index, double_index, label_8LL_batch, psi_phi_batch, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, target_tokens, target_seq, true_xyz = getData_PDB(domain, PDB_coords_seq_hdf5, esm2_3B_CPU, esm2_3B_GPU, batch_converter_esm2, device_esm, device1_spired, device2_spired, FLAGS.train_length, FLAGS)
            else:
                single_label_bacth, double_label_bacth, single_index, double_index, label_8LL_batch, psi_phi_batch, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, target_tokens, target_seq, true_xyz = getData_PDB(domain, CATH_coords_seq_hdf5, esm2_3B_CPU, esm2_3B_GPU, batch_converter_esm2, device_esm, device1_spired, device2_spired, FLAGS.train_length, FLAGS)

        s1 = time.time()
        t_dataload = s1 - s0

        L = label_8LL_batch.shape[-1]
        if L < 16:
            continue
        # frozen SPIRED or not
        if domain in fitness_sample_list and L > FLAGS.train_length:
            for name, param in union_model.named_parameters():
                if name.startswith("SPIRED"):
                    param.requires_grad = False
        else:
            for name, param in union_model.named_parameters():
                param.requires_grad = True

        ## training process
        single_pred, double_pred, Predxyz, PredCadistavg, Plddt, phi_psi_1D, soft_spearman_loss, all_spearman_corr, single_spearman_corr, double_spearman_corr, Loss_struct, loss_FU_dict, LossCE, LossCA, loss_phi_psi_1D = threadTrain(union_model, target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, single_index, double_index, single_label_bacth, double_label_bacth, label_8LL_batch, psi_phi_batch, FLAGS, i, device1_spired)

        LossDict = train_loss_dict(LossDict, Predxyz, loss_FU_dict, LossCE, LossCA, loss_phi_psi_1D)

        ## lDDT calculation
        truelddt = loss_FU_dict["truelddt"]["4th"]["fape40_2"]
        truelddt_np = np.mean(truelddt[0].cpu().detach().numpy(), axis=1)
        truelddt_median = np.median(truelddt_np)
        truelddt_max = np.max(truelddt_np)

        Loss_train_epoch["batch"].append(sample_index)
        Loss_train_epoch["sample_name"].append(domain)
        Loss_train_epoch["soft_spearman_loss"].append(soft_spearman_loss.item())
        Loss_train_epoch["all_spearman_corr"].append(all_spearman_corr.item())
        Loss_train_epoch["single_spearman_corr"].append(single_spearman_corr.item())
        Loss_train_epoch["double_spearman_corr"].append(double_spearman_corr.item())
        Loss_train_epoch["loss_struct"].append(Loss_struct.item())
        Loss_train_epoch["RD_loss_sum"].append(loss_FU_dict["optim"].item())
        Loss_train_epoch["loss_phi_psi_1D"].append(loss_phi_psi_1D.item())
        Loss_train_epoch["loss_CE"].append(LossCE["optim"].item())
        Loss_train_epoch["fape40_2"].append(loss_FU_dict["Fape"]["4th"]["fape40_2"].item())
        Loss_train_epoch["truelddt_median"].append(truelddt_median)
        Loss_train_epoch["truelddt_max"].append(truelddt_max)

        ## SUM Loss
        Loss_struct, soft_spearman_loss = (Loss_struct.to(device1_spired), soft_spearman_loss.to(device1_spired))

        if domain in fitness_sample_list:
            Loss_sum = FLAGS.weight_loss_struct * Loss_struct + FLAGS.weight_loss_spearman * soft_spearman_loss
        else:
            Loss_sum = FLAGS.weight_loss_struct * Loss_struct

        print("Loss_struct", Loss_struct.item(), "soft_spearman_loss", soft_spearman_loss.item(), "single_spearman_corr", single_spearman_corr.item(), "double_spearman_corr", double_spearman_corr.item(), "Loss_sum", Loss_sum.item())
        Loss_train_epoch["loss_sum"].append(Loss_sum.item())

        ## loss backward
        Loss_sum = Loss_sum / FLAGS.batch
        Loss_sum.backward()  # Loss backward
        optimizer.step()  # update parameters
        print("gradient updated.")
        optimizer.zero_grad()  # zero gradient

        s2 = time.time()
        t_train = s2 - s1
        print("fape40_2", loss_FU_dict["Fape"]["4th"]["fape40_2"].item())
        print("t_dataload, t_train", t_dataload, t_train)

    state = {"net": union_model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, os.path.join(FLAGS.model_dir, "SPIRED_Fitness_epoch{}.pth".format(epoch)))

    ## validation process
    union_model.eval()

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

    with torch.no_grad():
        valid_index = list(range(len(valid_set)))
        for i in valid_index:
            domain = valid_set[i]
            print("valid domain", domain)

            if domain in fitness_sample_list:
                single_label_bacth, double_label_bacth, single_index, double_index, label_8LL_batch, psi_phi_batch, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, target_tokens, target_seq, true_xyz = getData_fitness("test", domain, fit_coords_seq_hdf5, esm2_3B_hdf5, fitness_data_h5, batch_converter_esm2, device1_spired, device2_spired, FLAGS)
            else:
                single_label_bacth, double_label_bacth, single_index, double_index, label_8LL_batch, psi_phi_batch, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, target_tokens, target_seq, true_xyz = getData_PDB(domain, valid_coords_seq_hdf5, esm2_3B_CPU, esm2_3B_GPU, batch_converter_esm2, device_esm, device1_spired, device2_spired, 10000, FLAGS)

            #
            (single_pred, double_pred, Predxyz, PredCadistavg, Plddt, phi_psi_1D, soft_spearman_loss, all_spearman_corr, single_spearman_corr, double_spearman_corr, Loss_struct, loss_FU_dict, LossCE, LossCA, loss_phi_psi_1D) = threadValid(domain, union_model, target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, single_index, double_index, single_label_bacth, double_label_bacth, label_8LL_batch, psi_phi_batch, FLAGS, i, device1_spired)

            LossDict = valid_loss_dict(LossDict, Predxyz, loss_FU_dict, LossCE, LossCA, loss_phi_psi_1D)

            ## output into pt file
            saved_folder = os.path.join(FLAGS.model_dir, "pred_fitness")
            saved_folder_sample = os.path.join(saved_folder, domain)
            if not os.path.exists(saved_folder_sample):
                os.system("mkdir -p " + saved_folder_sample)
            torch.save(single_pred[0].detach().cpu().clone(), f"{saved_folder_sample}/{domain}_single.pt")
            torch.save(double_pred[0].detach().cpu().clone(), f"{saved_folder_sample}/{domain}_double.pt")

            ## lDDT calculation
            truelddt = loss_FU_dict["truelddt"]["4th"]["fape40_2"]  # (batch,L,L)
            truelddt_np = np.mean(truelddt[0].cpu().detach().numpy(), axis=1)  # (L,)
            truelddt_median = np.median(truelddt_np)
            truelddt_max = np.max(truelddt_np)
            fape40_2 = loss_FU_dict["Fape"]["4th"]["fape40_2"]

            Loss_struct, soft_spearman_loss = (Loss_struct.to(device1_spired), soft_spearman_loss.to(device1_spired))

            if domain in fitness_sample_list:
                Loss_sum = FLAGS.weight_loss_struct * Loss_struct + FLAGS.weight_loss_spearman * soft_spearman_loss
            else:
                Loss_sum = FLAGS.weight_loss_struct * Loss_struct

            print("Loss_struct", Loss_struct.item(), "soft_spearman_loss", soft_spearman_loss.item(), "single_spearman_corr", single_spearman_corr.item(), "double_spearman_corr", double_spearman_corr.item(), "Loss_sum", Loss_sum.item())

            if FLAGS.write_pdb > 0:
                Predxyz_4thBlock_6thlayer = Predxyz["4th"][-1]  # (batch=1,3,L,L)
                write_pdb(Predxyz_4thBlock_6thlayer[0], target_seq, 4, domain, FLAGS.model_dir)

            Loss_valid_epoch["batch"].append(i)
            Loss_valid_epoch["sample_name"].append(domain)
            Loss_valid_epoch["loss_sum"].append(Loss_sum.item())
            Loss_valid_epoch["soft_spearman_loss"].append(soft_spearman_loss.item())
            Loss_valid_epoch["all_spearman_corr"].append(all_spearman_corr.item())
            Loss_valid_epoch["single_spearman_corr"].append(single_spearman_corr.item())
            Loss_valid_epoch["double_spearman_corr"].append(double_spearman_corr.item())
            Loss_valid_epoch["loss_struct"].append(Loss_struct.item())
            Loss_valid_epoch["RD_loss_sum"].append(loss_FU_dict["optim"].item())
            Loss_valid_epoch["loss_phi_psi_1D"].append(loss_phi_psi_1D.item())
            Loss_valid_epoch["loss_CE"].append(LossCE["optim"].item())
            Loss_valid_epoch["fape40_2"].append(fape40_2.item())
            Loss_valid_epoch["truelddt_median"].append(truelddt_median)
            Loss_valid_epoch["truelddt_max"].append(truelddt_max)

            ## feature for Fitness Module
            xyz_LL3 = Predxyz["4th"][-1]  # (batch=1,3,L,L)
            plddt_value = Plddt["4th"][-1][0]  # (L, L)

    ## loss detail
    Loss_valid_all_epochs["epoch"].append(epoch)

    if FLAGS.valid_sample_num > 0:
        for key, loss_list in Loss_valid_epoch.items():
            if key not in ["batch", "sample_name"]:
                if re.search("spearman", key):
                    loss_list = [i for i in loss_list if i != 0]
                Loss_valid_all_epochs[key].append(np.nanmean(loss_list))
        write_loss_table(data_dict=Loss_valid_all_epochs, output_dir=FLAGS.model_dir, file_name="Loss_valid_mean_epoch{}.xls".format(epoch))
        write_loss_table(data_dict=Loss_valid_epoch, output_dir=FLAGS.model_dir, file_name="Loss_valid_epoch{}.xls".format(epoch))

    logLoss = [time.ctime(), epoch, lr]
    LossDictitem = list(LossDict.keys())
    for item0 in LossDictitem:
        logLoss.append(np.round(np.mean(LossDict[item0]), 4))
    makelog(logname=FLAGS.losslog, log=logLoss, commandline=sys.argv, header=["time", "epoch", "lr"] + LossDictitem)

    Loss_train_all_epochs["epoch"].append(epoch)
    if FLAGS.train_sample_num > 0:
        for key, loss_list in Loss_train_epoch.items():
            if key not in ["batch", "sample_name"]:
                if re.search("spearman", key):
                    loss_list = [i for i in loss_list if i != 0]
                Loss_train_all_epochs[key].append(np.nanmean(loss_list))
        write_loss_table(data_dict=Loss_train_all_epochs, output_dir=FLAGS.model_dir, file_name="Loss_train_mean_epoch{}.xls".format(epoch))
        write_loss_table(data_dict=Loss_train_epoch, output_dir=FLAGS.model_dir, file_name="Loss_train_epoch{}.xls".format(epoch))

    train_list = Loss_train_epoch["sample_name"]
    train_loss_log = {}
    train_loss_log["sample_name"] = train_list
    for item in LossDictitem:
        if re.search("train", item):
            train_loss_log[item] = LossDict[item]
    Loss_df = pd.DataFrame.from_dict(train_loss_log)
    Loss_df.to_csv("All_Loss_train_epoch{}.xls".format(epoch), sep="\t", na_rep="nan", index=False)

    valid_list = Loss_valid_epoch["sample_name"]
    valid_loss_log = {}
    valid_loss_log["sample_name"] = valid_list
    for item in LossDictitem:
        if re.search("valid", item):
            valid_loss_log[item] = LossDict[item]
            print(item, len(LossDict[item]))
    Loss_df = pd.DataFrame.from_dict(valid_loss_log)
    Loss_df.to_csv("All_Loss_valid_epoch{}_sample{}.xls".format(epoch, sample_index), sep="\t", na_rep="nan", index=False)


PDB_coords_seq_hdf5.close()
CATH_coords_seq_hdf5.close()
fit_coords_seq_hdf5.close()
valid_coords_seq_hdf5.close()
esm2_3B_hdf5.close()
