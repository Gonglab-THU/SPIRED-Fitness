import os, sys, argparse

env_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(env_path + "/../")
sys.path.append(env_path + "/utils/")

import torch
import pandas as pd
from model import PretrainModel
from utils.loss import spearman_loss
from utils.metrics import spearman_corr

parser = argparse.ArgumentParser()
parser.add_argument("--train_valid_list", type=str, default="./data/double_mut_train.csv", help="sample list file of training set")
parser.add_argument("--training_data", type=str, default="./", help="data for training set")
parser.add_argument("--outdir", type=str, default="./", help="data for training set")
FLAGS = parser.parse_args()
output_dir = FLAGS.outdir


class ProcessingData(torch.utils.data.Dataset):

    def __init__(self, train_csv, train_pt, dataset_type):

        self.train_csv = train_csv
        self.train_pt = train_pt
        self.dataset_type = dataset_type

    def __len__(self):

        return len(self.train_csv)

    def __getitem__(self, index):

        name = self.train_csv.iloc[index].name
        data = self.train_pt[name]
        return data, data["single_label"], data["double_label"], data["single_" + self.dataset_type], data["double_" + self.dataset_type], name, len(data["seq"])


def to_gpu(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [to_gpu(i, device=device) for i in obj]
    elif isinstance(obj, tuple):
        return (to_gpu(i, device=device) for i in obj)
    elif isinstance(obj, dict):
        return {i: to_gpu(j, device=device) for i, j in obj.items()}
    else:
        return obj


#######################################################################
# predifined parameters
#######################################################################

train_dataset_name = "dp|mavedb|k50"

# node_dims = [32]
# num_layers = [1, 2, 3, 4]
# n_heads = [4, 8]
# pair_dims = [16, 32, 64]
node_dims = [32]
num_layers = [2]
n_heads = [4, 8]
pair_dims = [16, 32, 64]

device = 0
seed = 0
early_stop = 20
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

#######################################################################
# data
#######################################################################

train_pt = torch.load(FLAGS.training_data)
train_csv = pd.read_csv(FLAGS.train_valid_list, index_col=0)
train_csv = train_csv[train_csv["train_dataset_name"].str.contains(train_dataset_name)]

train_dataset = ProcessingData(train_csv, train_pt, "train")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

validation_dataset = ProcessingData(train_csv, train_pt, "validation")
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

test_dataset = ProcessingData(train_csv, train_pt, "test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

#######################################################################
# train
#######################################################################

for node_dim in node_dims:
    for num_layer in num_layers:
        for n_head in n_heads:
            for pair_dim in pair_dims:
                file = f"{output_dir}/node_dim_{node_dim}-num_layer_{num_layer}-n_head_{n_head}-pair_dim_{pair_dim}"
                if not os.path.exists(f"{file}/pred.csv"):
                    os.system(f"mkdir -p {file}")

                    model = PretrainModel(node_dim, n_head, pair_dim, num_layer).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5, verbose=True)

                    best_loss = float("inf")
                    stop_step = 0
                    loss_csv = pd.DataFrame()
                    for epoch in range(1000):

                        # train
                        model.train()
                        epoch_loss = 0
                        epoch_corr = 0
                        for data, single_label, double_label, single_index, double_index, _, length in train_loader:
                            if length < 800:
                                data, single_label, double_label = to_gpu(data, device), to_gpu(single_label, device), to_gpu(double_label, device)
                                model.to(device)
                            else:
                                model = model.to("cpu")

                            # renew optimizer
                            optimizer_new = torch.optim.Adam(model.parameters())
                            optimizer_new.load_state_dict(optimizer.state_dict())
                            optimizer = optimizer_new
                            optimizer.zero_grad()
                            single_pred, double_pred = model(data)

                            if len(double_index[0]) != 0:
                                pred = torch.cat((single_pred[0][single_index], double_pred[0][double_index]), dim=0)
                                label = torch.cat((single_label[0][single_index], double_label[0][double_index]), dim=0)
                                loss = spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), 1e-2, "kl")
                                epoch_corr += spearman_corr(pred, label).item()
                            else:
                                loss = spearman_loss(single_pred[0][single_index].unsqueeze(0), single_label[0][single_index].unsqueeze(0), 1e-2, "kl")
                                epoch_corr += spearman_corr(single_pred[0][single_index], single_label[0][single_index]).item()
                            epoch_loss += loss.item()

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type=2, max_norm=10, error_if_nonfinite=True)
                            optimizer.step()

                        loss_csv.loc[epoch, "train_loss"] = epoch_loss / len(train_loader)
                        loss_csv.loc[epoch, "train_corr"] = epoch_corr / len(train_loader)

                        # validation
                        model.eval()
                        epoch_loss = 0
                        epoch_single_loss = 0
                        epoch_double_loss = 0
                        double_epochs = 0
                        with torch.no_grad():
                            for data, single_label, double_label, single_index, double_index, _, length in validation_loader:
                                if length < 800:
                                    data, single_label, double_label = to_gpu(data, device), to_gpu(single_label, device), to_gpu(double_label, device)
                                    model.to(device)
                                else:
                                    model = model.to("cpu")
                                single_pred, double_pred = model(data)

                                epoch_single_loss += -spearman_corr(single_pred[0][single_index], single_label[0][single_index]).item()
                                if len(double_index[0]) != 0:
                                    pred = torch.cat((single_pred[0][single_index], double_pred[0][double_index]), dim=0)
                                    label = torch.cat((single_label[0][single_index], double_label[0][double_index]), dim=0)
                                    epoch_loss += -spearman_corr(pred, label).item()
                                    epoch_double_loss += -spearman_corr(double_pred[0][double_index], double_label[0][double_index]).item()
                                    double_epochs += 1
                                else:
                                    epoch_loss += -spearman_corr(single_pred[0][single_index], single_label[0][single_index]).item()

                        validation_loss = epoch_loss / len(validation_loader)
                        loss_csv.loc[epoch, "validation_loss"] = validation_loss
                        loss_csv.loc[epoch, "single_validation_loss"] = epoch_single_loss / len(validation_loader)
                        loss_csv.loc[epoch, "double_validation_loss"] = epoch_double_loss / double_epochs

                        print(loss_csv)
                        torch.save(model, f"{file}/{epoch}.pt")

                        scheduler.step(validation_loss)
                        if validation_loss < best_loss:
                            stop_step = 0
                            best_loss = validation_loss
                            torch.save(model, f"{file}/best.pt")
                        else:
                            stop_step += 1
                            if stop_step >= early_stop:
                                break
                        loss_csv.to_csv(f"{file}/loss.csv")

                    test_csv = pd.DataFrame(index=train_pt.keys())
                    model = torch.load(f"{file}/best.pt", map_location=lambda storage, loc: storage.cuda(device))
                    model.eval()

                    # save torchscript model
                    length = 10
                    torchscript_model = torch.jit.trace(model.to("cpu"), {"1d": torch.randn(1, length, 1280), "3d": torch.randn(1, length, length, 3), "plddt": torch.randn(1, length, length), "single_logits": torch.randn(1, 5, length, 20), "double_logits": torch.randn(1, 5, length, length, 400)})
                    torch.jit.save(torchscript_model, f"{file}/best.jit")

                    # test
                    model.eval()
                    corr_dict = {}
                    with torch.no_grad():
                        for data, single_label, double_label, single_index, double_index, name, length in test_loader:
                            if length < 800:
                                data, single_label, double_label = to_gpu(data, device), to_gpu(single_label, device), to_gpu(double_label, device)
                                model.to(device)
                            else:
                                model = model.to("cpu")
                            single_pred, double_pred = model(data)
                            test_csv.loc[name[0], "single_corr"] = spearman_corr(single_pred[0][single_index], single_label[0][single_index]).item()
                            if len(double_index[0]) != 0:
                                pred = torch.cat((single_pred[0][single_index], double_pred[0][double_index]), dim=0)
                                label = torch.cat((single_label[0][single_index], double_label[0][double_index]), dim=0)
                                test_csv.loc[name[0], "double_corr"] = spearman_corr(double_pred[0][double_index], double_label[0][double_index]).item()
                                test_csv.loc[name[0], "all_corr"] = spearman_corr(pred, label).item()
                            else:
                                test_csv.loc[name[0], "all_corr"] = spearman_corr(single_pred[0][single_index], single_label[0][single_index]).item()
                    test_csv.to_csv(f"{file}/pred.csv")
