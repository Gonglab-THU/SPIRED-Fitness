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
from seq2struct.Model import SPIRED_Model
from seq2struct.utils_train_valid import threadTrain, threadValid, getDataTrain, getDataValid, train_loss_dict, valid_loss_dict, make_loss_dict, write_pdb, makelog, load_model_parameters, write_loss_table


parser = argparse.ArgumentParser()
parser.add_argument("--PDB_coords_seq_hdf5", type=str, default="./", help="hdf5 file path for PDB data of train set")
# parser.add_argument("--CATH_coords_seq_hdf5", type=str, default="./", help="hdf5 file path for CATH data of train set")
parser.add_argument("--valid_coords_seq_hdf5", type=str, default="./", help="valid hdf5 path for coords and seqs data of validation set")
parser.add_argument("--train_list", type=str, default="./", help="path for train sample list file")
parser.add_argument("--valid_list", type=str, default="./", help="path for validation sample list file")
parser.add_argument("--maxlen", type=int, default=256, help="max length of protein sequences when training SPIRED model, larger protein will be cropped")
parser.add_argument("--valid_length", type=int, default=600, help="max lehgth of validation protein sequences, larger protein will be cropped")
parser.add_argument("--train_cycle", type=int, default=4, help="max cycle number for train samples")
parser.add_argument("--valid_cycle", type=int, default=4, help="cycle number for validation samples")
parser.add_argument("--batch", type=int, default=64, help="batch for Gradient Accumulationn")
parser.add_argument("--ESM_length_cutoff", type=int, default=600, help="max lehgth of protein sequence for ESM2 model")
parser.add_argument("--epoch_start", type=int, default=1, help="number of start point of training epochs")
parser.add_argument("--epoch_end", type=int, default=30, help="number of end point of training epochs")
parser.add_argument("--lr_start", type=float, default=0.001, help="learning rate, default=0.001")
parser.add_argument("--lr_end", type=float, default=0.001, help="learning rate, default=0.001")
parser.add_argument("--gpu_SPIRED", type=int, default=[0, 1], nargs="+", help="the ID list of GPUs used for SPIRED model")
parser.add_argument("--gpu_esm", type=int, default=[0], nargs="+", help="the ID list of GPUs used for ESM2 model")
parser.add_argument("--SPIRED_saved_model", type=str, default="None", help="path of saved parameters of SPIRED model")
parser.add_argument("--train_sample_num", type=int, help="sample (mini-batch) number for training set")
parser.add_argument("--valid_sample_num", type=int, help="sample number for validation set")
parser.add_argument("--out_dir", type=str, default="./", help="Directory for saving models")
parser.add_argument("--losslog", type=str, default="struct_loss_log.xls", help="path for struct loss log")
parser.add_argument("--write_pdb_num", type=int, default=4, help="path for predicted (output) pdb files")
FLAGS = parser.parse_args()


## load data
PDB_coords_seq_hdf5 = h5py.File(FLAGS.PDB_coords_seq_hdf5, "r")
# CATH_coords_seq_hdf5 = h5py.File(FLAGS.CATH_coords_seq_hdf5, 'r')
valid_coords_seq_hdf5 = h5py.File(FLAGS.valid_coords_seq_hdf5, "r")

train_set, valid_set = ([], [])

with open(FLAGS.train_list, "r") as train_list:
    for line in train_list:
        sample_name = re.split("\t", line.strip())[0]
        train_set.append(sample_name)

with open(FLAGS.valid_list, "r") as valid_list:
    for line in valid_list:
        sample_name = re.split("\t", line.strip())[0]
        valid_set.append(sample_name)

train_set = train_set[: FLAGS.train_sample_num]
valid_set = valid_set[: FLAGS.valid_sample_num]


## GPU device
device_list_SPIRED = ["cuda:" + str(gpu) for gpu in FLAGS.gpu_SPIRED]
device_list_esm = ["cuda:" + str(gpu) for gpu in FLAGS.gpu_esm]

device1_SPIRED = torch.device(device_list_SPIRED[0])
device2_SPIRED = torch.device(device_list_SPIRED[-1])
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


SPIRED_Model = SPIRED_Model(depth=2, channel=128, device_list=device_list_SPIRED)

## load the saved model parameters
SPIRED_Model = load_model_parameters(SPIRED_Model, FLAGS.SPIRED_saved_model)

## Load ESM-2 model
esm_model_path = "esm2/esm2_t36_3B_UR50D.pt"
esm2_CPU, alphabet_esm2 = esm.pretrained.load_model_and_alphabet(esm_model_path)
esm2_GPU, alphabet_esm2 = esm.pretrained.load_model_and_alphabet(esm_model_path)
batch_converter_esm2 = alphabet_esm2.get_batch_converter()
esm2_GPU = esm2_GPU.to(device=device_esm)
esm2_CPU.eval()
esm2_GPU.eval()

# calculate number of parameters of the model
params_SPIRED = sum(p.numel() for p in list(SPIRED_Model.parameters())) / 1e6  # numel()
print("Parameters of SPIRED_Model Model: %.3fM" % (params_SPIRED))

## initialize the optimizer
optimizer_SPIRED = torch.optim.Adam(SPIRED_Model.parameters(), lr=FLAGS.lr_start, betas=(0.9, 0.999))


def create_loss_dict(sample_name=False):
    dict = {
        "epoch": [],
        "batch": [],
        "sample_name": [],
        "loss_sum": [],
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


def warmup_learning_rate(optimizer, lr_end):
    lr = optimizer.param_groups[0]["lr"]
    if lr < lr_end:
        lr = lr + 0.001 * lr_end
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    print("learning rate", optimizer.param_groups[0]["lr"])
    return optimizer, lr


Loss_train_all_epochs, Loss_valid_all_epochs = (create_loss_dict(sample_name=False), create_loss_dict(sample_name=False))

# train and validation process
lr = FLAGS.lr_start
for epoch in range(FLAGS.epoch_start, FLAGS.epoch_end + 1):

    Loss_train_epoch, Loss_valid_epoch = (create_loss_dict(sample_name=True), create_loss_dict(sample_name=True))

    # shuffle train data set
    random.shuffle(train_set)

    ## set SPIRED into training mode
    SPIRED_Model.train()

    sample_index = 0
    LossDict = make_loss_dict()

    ## training
    for i in range(0, FLAGS.train_sample_num):
        sample_index += 1

        protein_name = train_set[i]
        protein_name_batch = re.split(",", protein_name)
        print("train_sample", protein_name)

        s0 = time.time()
        (label_batch, psi_phi_batch, f1d_batch, f2d_batch, target_tokens, true_xyz) = getDataTrain(protein_name_batch, PDB_coords_seq_hdf5, FLAGS.maxlen, esm2_CPU, esm2_GPU, batch_converter_esm2, device_esm, device1_SPIRED, device1_SPIRED, FLAGS)

        # print("f1d_batch",f1d_batch.shape, "f2d_batch",f2d_batch.shape, "label_batch",label_batch.shape)
        s1 = time.time()
        t_dataload = s1 - s0

        L = label_batch.shape[-1]
        if L < 40:
            continue

        ##-------------------------------------------------------------
        ## SPIRED predicts structures, calculates loss and updates parameters
        ##-------------------------------------------------------------

        Predxyz, PredCadistavg, Plddt, phi_psi_1D, Loss_sum, loss_struct_dict, LossCE, LossCA, loss_phi_psi_1D = threadTrain(SPIRED_Model, target_tokens, f1d_batch, f2d_batch, label_batch, psi_phi_batch, FLAGS, i)

        LossDict = train_loss_dict(LossDict, Predxyz, loss_struct_dict, LossCE, LossCA, loss_phi_psi_1D)

        ## lDDT calculation
        truelddt = loss_struct_dict["truelddt"]["4th"]["fape40_2"]  # (batch,L,L)
        loss_truelddt = truelddt.mean()
        truelddt_np = np.mean(truelddt[0].cpu().detach().numpy(), axis=1)  # (L,)
        truelddt_median = np.median(truelddt_np)
        truelddt_max = np.max(truelddt_np)

        Loss_train_epoch["batch"].append(sample_index)
        Loss_train_epoch["sample_name"].append(protein_name)
        Loss_train_epoch["loss_sum"].append(Loss_sum.item())
        Loss_train_epoch["RD_loss_sum"].append(loss_struct_dict["optim"].item())
        Loss_train_epoch["loss_phi_psi_1D"].append(loss_phi_psi_1D.item())
        Loss_train_epoch["loss_CE"].append(LossCE["optim"].item())
        Loss_train_epoch["fape40_2"].append(loss_struct_dict["Fape"]["4th"]["fape40_2"].item())
        Loss_train_epoch["truelddt_median"].append(truelddt_median)
        Loss_train_epoch["truelddt_max"].append(truelddt_max)

        ## loss backward
        Loss_sum = Loss_sum / FLAGS.batch  # Loss Normlization for Gradient Accumulation
        Loss_sum.backward()  # calculate gradient

        s2 = time.time()
        t_train = s2 - s1
        # print("fape40_2", loss_struct_dict['Fape']['4th']['fape40_2'].item())
        print("time_dataload, time_train", round(t_dataload, 2), round(t_train, 2))

        if sample_index % FLAGS.batch == 0 or sample_index == FLAGS.train_sample_num:

            optimizer_SPIRED.step()  # update parameters
            print("gradient updated.")

            optimizer_SPIRED.zero_grad()  # clear gradient

            ## warming up learning rate
            optimizer_SPIRED, lr = warmup_learning_rate(optimizer_SPIRED, FLAGS.lr_end)

        ## validation and save the model parameters
        if sample_index % 30000 == 0 or sample_index == FLAGS.train_sample_num:
            torch.save(SPIRED_Model.state_dict(), os.path.join(FLAGS.out_dir, "SPIRED_Model_Para_epoch{}_sample{}.pth".format(epoch, sample_index)))
            torch.cuda.empty_cache()

            ## validation
            SPIRED_Model.eval()

            with torch.no_grad():  # no gradient when validation
                valid_index = list(range(len(valid_set)))
                for i in valid_index:
                    protein_name = valid_set[i]
                    print("valid protein_name", protein_name)

                    labelall, psi_phi, f1d, f2d, target_tokens, target_seq, true_xyz = getDataValid(protein_name, valid_coords_seq_hdf5, esm2_CPU, esm2_GPU, batch_converter_esm2, device_esm, device1_SPIRED, device2_SPIRED, FLAGS)

                    L = labelall.shape[-1]
                    if L < 40:
                        continue

                    Predxyz, PredCadistavg, Plddt, phi_psi_1D, loss_sum, loss_struct_dict, LossCE, LossCA, loss_phi_psi_1D = threadValid(protein_name, SPIRED_Model, target_tokens, f1d, f2d, labelall, psi_phi, FLAGS, i)

                    LossDict = valid_loss_dict(LossDict, Predxyz, loss_struct_dict, LossCE, LossCA, loss_phi_psi_1D)

                    truelddt = loss_struct_dict["truelddt"]["4th"]["fape40_2"]  # (batch,L,L)
                    truelddt_np = np.mean(truelddt[0].cpu().detach().numpy(), axis=1)  # (L,)
                    truelddt_median = np.median(truelddt_np)
                    truelddt_max = np.max(truelddt_np)
                    fape40_2 = loss_struct_dict["Fape"]["4th"]["fape40_2"]

                    if FLAGS.write_pdb_num > 0:
                        plddt_value = Plddt["4th"][-1][0]  # (L, L)
                        plddt_value_L = torch.mean(plddt_value, dim=1)  # (L,)
                        plddt_topK_idx = torch.topk(plddt_value_L, FLAGS.write_pdb_num)[-1]
                        xyz_topK = Predxyz["4th"][-1][0, :, plddt_topK_idx, :].permute(1, 2, 0)  # (8,L,3)
                        write_pdb(xyz_topK, target_seq, FLAGS.write_pdb_num, protein_name, FLAGS.out_dir)

                    Loss_valid_epoch["batch"].append(i)
                    Loss_valid_epoch["sample_name"].append(protein_name)
                    Loss_valid_epoch["loss_sum"].append(loss_sum.item())
                    Loss_valid_epoch["RD_loss_sum"].append(loss_struct_dict["optim"].item())
                    Loss_valid_epoch["loss_phi_psi_1D"].append(loss_phi_psi_1D.item())
                    Loss_valid_epoch["loss_CE"].append(LossCE["optim"].item())
                    Loss_valid_epoch["fape40_2"].append(fape40_2.item())
                    Loss_valid_epoch["truelddt_median"].append(truelddt_median)
                    Loss_valid_epoch["truelddt_max"].append(truelddt_max)

            SPIRED_Model.train()

            ## record loss detail write to file
            Loss_train_all_epochs["epoch"].append(epoch)
            Loss_valid_all_epochs["epoch"].append(epoch)

            if FLAGS.train_sample_num > 0:
                for key, loss_list in Loss_train_epoch.items():
                    if key not in ["batch", "sample_name"]:
                        print("key", key, len(loss_list))
                        Loss_train_all_epochs[key].append(np.nanmean(loss_list))
                write_loss_table(data_dict=Loss_train_all_epochs, output_dir=FLAGS.out_dir, file_name="Loss_train_mean_epoch{}.xls".format(epoch))
                write_loss_table(data_dict=Loss_train_epoch, output_dir=FLAGS.out_dir, file_name="Loss_train_epoch{}.xls".format(epoch))

            if FLAGS.valid_sample_num > 0:
                for key, loss_list in Loss_valid_epoch.items():
                    if key not in ["batch", "sample_name"]:
                        print("key", key, len(loss_list))
                        Loss_valid_all_epochs[key].append(np.nanmean(loss_list))
                write_loss_table(data_dict=Loss_valid_all_epochs, output_dir=FLAGS.out_dir, file_name="Loss_valid_mean_epoch{}_sample{}.xls".format(epoch, sample_index))
                write_loss_table(data_dict=Loss_valid_epoch, output_dir=FLAGS.out_dir, file_name="Loss_valid_epoch{}_sample{}.xls".format(epoch, sample_index))

            logLoss = [time.ctime(), epoch, lr]
            LossDictitem = list(LossDict.keys())
            for item0 in LossDictitem:
                logLoss.append(np.round(np.mean(LossDict[item0]), 4))
            makelog(logname=os.path.join(FLAGS.out_dir, FLAGS.losslog), log=logLoss, commandline=sys.argv, header=["time", "epoch", "lr"] + LossDictitem)

            train_list = Loss_train_epoch["sample_name"]
            train_loss_log = {}
            train_loss_log["sample_name"] = train_list
            for item in LossDictitem:
                if re.search("train", item):
                    train_loss_log[item] = LossDict[item]
            Loss_df = pd.DataFrame.from_dict(train_loss_log)
            Loss_df.to_csv(os.path.join(FLAGS.out_dir, "All_Loss_train_epoch{}.xls").format(epoch), sep="\t", na_rep="nan", index=False)

            valid_list = Loss_valid_epoch["sample_name"]
            valid_loss_log = {}
            valid_loss_log["sample_name"] = valid_list
            for item in LossDictitem:
                if re.search("valid", item):
                    valid_loss_log[item] = LossDict[item]
                    # print(item, len(LossDict[item]))
            Loss_df = pd.DataFrame.from_dict(valid_loss_log)
            Loss_df.to_csv(os.path.join(FLAGS.out_dir, "All_Loss_valid_epoch{}_sample{}.xls").format(epoch, sample_index), sep="\t", na_rep="nan", index=False)


PDB_coords_seq_hdf5.close()
# CATH_coords_seq_hdf5.close()
valid_coords_seq_hdf5.close()
