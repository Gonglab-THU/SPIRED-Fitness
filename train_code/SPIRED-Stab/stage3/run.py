import torch
import pandas as pd
from model import Model
from data_ddG_dTm import ProcessingData, train_mse_model, validation_mse_model, test_model, convert_batch_data

#######################################################################
# predifined parameters
#######################################################################

batch_size = 4
node_dim = 32
num_layer = 3
n_head = 8
pair_dim = 64

device = 0
seed = 0
early_stop = 20
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

#######################################################################
# data
#######################################################################

data = pd.read_csv("ddG_dTm_data.csv")
data["protein_name"] = data["mut_name"].str.rsplit("_", n=1).str[0]
data = data[data["dataset"].str.contains("ddG_train|dTm_train")]
data.loc[:, "length"] = data["wt_seq"].str.len()
# data = data[data["length"] <= 400]
all_data = data.copy()

train_csv = all_data.sample(frac=0.8, random_state=seed).copy()
validation_csv = all_data.drop(train_csv.index).copy()

train_csv["counts"] = train_csv.groupby("protein_name")["protein_name"].transform("count")
train_csv = train_csv[train_csv["counts"] >= batch_size]
train_csv.set_index("protein_name", inplace=True)
validation_csv["counts"] = validation_csv.groupby("protein_name")["protein_name"].transform("count")
validation_csv = validation_csv[validation_csv["counts"] >= batch_size]
validation_csv.set_index("protein_name", inplace=True)

validation_dataset = ProcessingData(convert_batch_data(validation_csv, batch_size, seed))
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

test_csv = pd.read_csv("ddG_dTm_test_data.csv")
test_dataset = ProcessingData(test_csv)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

#######################################################################
# train
#######################################################################

model = Model(node_dim, n_head, pair_dim, num_layer).to(device)
model.load_state_dict(torch.load(f"../stage2/all_model/node_dim_{node_dim}-num_layer_{num_layer}-n_head_{n_head}-pair_dim_{pair_dim}/best.pt", map_location="cpu").state_dict(), strict=False)

# ddG
# fixed pretrain parameters
for name, param in model.named_parameters():
    if "finetune_ddG_coef" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5, verbose=True)

best_loss = float("inf")
stop_step = 0
loss = pd.DataFrame()
for epoch in range(500):
    train_dataset = ProcessingData(convert_batch_data(train_csv, batch_size, epoch))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    train_loss = train_mse_model(model, optimizer, train_loader, "ddG")
    validation_ddG_loss, validation_dTm_loss = validation_mse_model(model, validation_loader)
    validation_loss = validation_ddG_loss
    loss.loc[f"{epoch}", "train_loss"] = train_loss
    loss.loc[f"{epoch}", "validation_loss"] = validation_ddG_loss

    print(loss)

    scheduler.step(validation_loss)
    if validation_loss < best_loss:
        stop_step = 0
        best_loss = validation_loss
        torch.save(model, "best.pt")
    else:
        stop_step += 1
        if stop_step >= early_stop:
            break
    loss.to_csv("ddG_loss.csv")

model = torch.load("best.pt", map_location=lambda storage, loc: storage.cuda(device))

# dTm
# fixed pretrain parameters
for name, param in model.named_parameters():
    if "finetune_dTm_coef" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5, verbose=True)

best_loss = float("inf")
stop_step = 0
loss = pd.DataFrame()
for epoch in range(500):
    train_dataset = ProcessingData(convert_batch_data(train_csv, batch_size, epoch))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    train_loss = train_mse_model(model, optimizer, train_loader, "dTm")
    validation_ddG_loss, validation_dTm_loss = validation_mse_model(model, validation_loader)
    validation_loss = validation_dTm_loss
    loss.loc[f"{epoch}", "train_loss"] = train_loss
    loss.loc[f"{epoch}", "validation_loss"] = validation_dTm_loss

    print(loss)

    scheduler.step(validation_loss)
    if validation_loss < best_loss:
        stop_step = 0
        best_loss = validation_loss
        torch.save(model, "best.pt")
    else:
        stop_step += 1
        if stop_step >= early_stop:
            break
    loss.to_csv("dTm_loss.csv")

model = torch.load("best.pt", map_location=lambda storage, loc: storage.cuda(device))
model.eval()
model = torch.load("best.pt")
result = test_model(model, test_loader, test_csv)
result.to_csv("pred.csv")
torch.save(model.state_dict(), "best.pth")
