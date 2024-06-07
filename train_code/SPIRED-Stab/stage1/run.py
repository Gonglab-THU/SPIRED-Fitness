import os
import torch
import pandas as pd
from model import Model
from data import ProcessingData, train_model, validation_model, test_model, convert_batch_data

#######################################################################
# predifined parameters
#######################################################################

batch_size = 64
dataset_name = "k50"

node_dims = [32]
num_layers = [1, 2, 3, 4]
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

wt_data_features = torch.load("wt_data_for_value_function.pt")
data = pd.read_pickle("data_for_value_function.pkl")
data = data[data["dataset_name"].str.contains(dataset_name)]

# data = data[data["length"] <= 500]

data = data[data["counts"] >= batch_size]
data.drop(["dataset_name", "counts", "length"], axis=1, inplace=True)

train_csv = data[data["cv_type"] == "train"].drop("cv_type", axis=1)
validation_csv = data[data["cv_type"] == "validation"].drop("cv_type", axis=1)
test_csv = data[data["cv_type"] == "test"].drop("cv_type", axis=1)

validation_dataset = ProcessingData(convert_batch_data(validation_csv, batch_size, seed), wt_data_features)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

test_dataset = ProcessingData(convert_batch_data(test_csv, batch_size, seed), wt_data_features)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#######################################################################
# train
#######################################################################

for node_dim in node_dims:
    for num_layer in num_layers:
        for n_head in n_heads:
            for pair_dim in pair_dims:
                file = f"all_model/node_dim_{node_dim}-num_layer_{num_layer}-n_head_{n_head}-pair_dim_{pair_dim}"
                if not os.path.exists(f"{file}/pred.csv"):
                    os.system(f"mkdir -p {file}")
                    model = Model(node_dim, n_head, pair_dim, num_layer).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5, verbose=True)

                    best_loss = float("inf")
                    stop_step = 0
                    loss = pd.DataFrame()
                    for epoch in range(500):

                        train_dataset = ProcessingData(convert_batch_data(train_csv, batch_size, epoch), wt_data_features)
                        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

                        train_loss, train_corr = train_model(model, optimizer, train_loader)
                        validation_loss = validation_model(model, validation_loader)

                        loss.loc[epoch, "train_loss"] = train_loss
                        loss.loc[epoch, "train_corr"] = train_corr
                        loss.loc[epoch, "validation_loss"] = validation_loss

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

                    model = torch.load(f"{file}/best.pt", map_location=lambda storage, loc: storage.cuda(device))
                    model.eval()
                    result = test_model(model, test_loader)
                    result.to_csv(f"{file}/pred.csv")
