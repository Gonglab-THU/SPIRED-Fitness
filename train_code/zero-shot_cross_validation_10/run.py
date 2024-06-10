import os
import torch
import random
import pandas as pd
from model import PretrainModel
from dms import ProcessingData, train_model, validation_model, test_model

#######################################################################
# predifined parameters
#######################################################################

node_dim = 32
num_layer = 2
n_head = 8
pair_dim = 32

device = 0
seed = 0
early_stop = 10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

#######################################################################
# data
#######################################################################

all_protein_names = ["A0A1I9GEU1_NEIME_Kennouche_2019", "A0A247D711_LISMN_Stadelmann_2021", "A4D664_9INFA_Soh_2019", "A4_HUMAN_Seuma_2022", "AACC1_PSEAI_Dandage_2018", "ADRB2_HUMAN_Jones_2020", "AICDA_HUMAN_Gajula_2014_3cycles", "ANCSZ_Hobbs_2022", "B2L11_HUMAN_Dutta_2010_binding-Mcl-1", "C6KNH7_9INFA_Lee_2018", "CAPSD_AAV2S_Sinai_2021", "CASP3_HUMAN_Roychowdhury_2020", "CASP7_HUMAN_Roychowdhury_2020", "CD19_HUMAN_Klesmith_2019_FMC_singles", "D7PM05_CLYGR_Somermeyer_2022", "ENVZ_ECOLI_Ghose_2023", "ESTA_BACSU_Nutschel_2020", "F7YBW7_MESOW_Ding_2023", "F7YBW8_MESOW_Aakre_2015", "GCN4_YEAST_Staller_2018", "GLPA_HUMAN_Elazar_2016", "GRB2_HUMAN_Faure_2021", "HEM3_HUMAN_Loggerenberg_2023", "KCNE1_HUMAN_Muhammad_2023_expression", "KCNJ2_MOUSE_Coyote-Maestas_2022_function", "LYAM1_HUMAN_Elazar_2016", "MET_HUMAN_Estevam_2023", "MLAC_ECOLI_MacRae_2023", "NRAM_I33A0_Jiang_2016", "OTC_HUMAN_Lo_2023", "OXDA_RHOTO_Vanella_2023_expression", "PAI1_HUMAN_Huttinger_2021", "PHOT_CHLRE_Chen_2023", "PPARG_HUMAN_Majithia_2016", "PPM1D_HUMAN_Miller_2022", "PRKN_HUMAN_Clausen_2023", "Q53Z42_HUMAN_McShan_2019_expression", "Q6WV13_9MAXI_Somermeyer_2022", "Q837P4_ENTFA_Meier_2023", "Q837P5_ENTFA_Meier_2023", "R1AB_SARS2_Flynn_2022", "RDRP_I33A0_Li_2023", "REV_HV1H2_Fernandes_2016", "RNC_ECOLI_Weeks_2023", "RPC1_LAMBD_Li_2019_low-expression", "S22A1_HUMAN_Yee_2023_activity", "SC6A4_HUMAN_Young_2021", "SERC_HUMAN_Xie_2023", "SHOC2_HUMAN_Kwon_2022", "TAT_HV1BR_Fernandes_2016"]

train_pt = torch.load("double_mut_data_50.pt")
cluster_csv = pd.read_csv("cluster.csv")
for k_fold_index in range(10):
    train_names = []
    test_names = []
    for index in cluster_csv.index:
        if cluster_csv.loc[index, "cluster_index"] - 1 == k_fold_index:
            test_names.append(cluster_csv.loc[index, "protein_name"])
        else:
            train_names.append(cluster_csv.loc[index, "protein_name"])
    train_csv = pd.DataFrame(index=train_pt.keys())

    random.seed(k_fold_index)
    validation_names = [random.choice(train_names)]
    train_names.remove(validation_names[0])

    train_dataset = ProcessingData(train_csv.loc[train_names], train_pt)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    validation_dataset = ProcessingData(train_csv.loc[validation_names], train_pt)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

    test_dataset = ProcessingData(train_csv.loc[test_names], train_pt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # train
    file = f"{k_fold_index}-fold"
    os.system(f"mkdir -p {file}")

    model = PretrainModel(node_dim, n_head, pair_dim, num_layer).to(device)
    pretrain_model_state_dict = torch.load("model_pth/SPIRED-Fitness.pth", map_location="cpu").copy()
    model_state_dict = model.state_dict()
    pretrain_model_state_dict = {k.split(".", 1)[-1]: v for k, v in pretrain_model_state_dict.items() if k.startswith("Fitness.")}
    model_state_dict.update(pretrain_model_state_dict)
    model.load_state_dict(model_state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5, verbose=True)

    best_loss = float("inf")
    stop_step = 0
    loss = pd.DataFrame()
    for epoch in range(1000):

        train_loss, train_corr = train_model(model, optimizer, train_loader)
        validation_loss = validation_model(model, validation_loader)
        test_loss = validation_model(model, test_loader)

        loss.loc[epoch, "train_loss"] = train_loss
        loss.loc[epoch, "train_corr"] = train_corr
        loss.loc[epoch, "validation_loss"] = validation_loss
        loss.loc[epoch, "test_corr"] = -test_loss

        print(loss)

        scheduler.step(validation_loss)
        if validation_loss < best_loss:
            stop_step = 0
            best_loss = validation_loss
            torch.save(model, f"{file}/best.pt")
        else:
            stop_step += 1
            if stop_step >= early_stop:
                break
        loss.to_csv(f"{file}/loss.csv")

    test_csv = pd.DataFrame(index=train_pt.keys())
    model = torch.load(f"{file}/best.pt", map_location=lambda storage, loc: storage.cuda(device))
    model.eval()

    single_corr_dict, double_corr_dict, all_corr_dict = test_model(model, test_loader)
    test_csv["single_corr"] = test_csv.index.map(single_corr_dict)
    if len(double_corr_dict) != 0:
        test_csv["double_corr"] = test_csv.index.map(double_corr_dict)
    test_csv["all_corr"] = test_csv.index.map(all_corr_dict)
    test_csv.to_csv(f"{file}/pred.csv")
