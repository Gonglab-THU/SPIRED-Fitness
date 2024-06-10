import torch
from loss import spearman_loss
from metrics import spearman_corr


class ProcessingData(torch.utils.data.Dataset):

    def __init__(self, train_csv, train_pt):

        self.train_csv = train_csv
        self.train_pt = train_pt

    def __len__(self):

        return len(self.train_csv)

    def __getitem__(self, index):

        name = self.train_csv.iloc[index].name
        data = self.train_pt[name]
        return data, data["single_label"], data["double_label"], name


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


def train_model(model, optimizer, loader):

    model.train()
    device = next(model.parameters()).device
    epoch_loss = 0
    epoch_corr = 0
    for data, single_label, double_label, _ in loader:
        data, single_label, double_label = to_gpu(data, device), to_gpu(single_label, device), to_gpu(double_label, device)
        optimizer.zero_grad()
        single_pred, double_pred = model(data)

        single_index = torch.where(~torch.isnan(single_label))
        double_index = torch.where(~torch.isnan(double_label))
        if len(double_index[0]) != 0:
            pred = torch.cat((single_pred[single_index], double_pred[double_index]), dim=0)
            label = torch.cat((single_label[single_index], double_label[double_index]), dim=0)
            loss = spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), 1e-2, "kl")
            epoch_corr += spearman_corr(pred, label).item()
        else:
            loss = spearman_loss(single_pred[single_index].unsqueeze(0), single_label[single_index].unsqueeze(0), 1e-2, "kl")
            epoch_corr += spearman_corr(single_pred[single_index], single_label[single_index]).item()
        epoch_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type=2, max_norm=10, error_if_nonfinite=True)
        optimizer.step()
    return epoch_loss / len(loader), epoch_corr / len(loader)


def validation_model(model, loader):

    model.eval()
    device = next(model.parameters()).device
    epoch_loss = 0
    with torch.no_grad():
        for data, single_label, double_label, _ in loader:
            data, single_label, double_label = to_gpu(data, device), to_gpu(single_label, device), to_gpu(double_label, device)
            single_pred, double_pred = model(data)

            single_index = torch.where(~torch.isnan(single_label))
            double_index = torch.where(~torch.isnan(double_label))
            if len(double_index[0]) != 0:
                pred = torch.cat((single_pred[single_index], double_pred[double_index]), dim=0)
                label = torch.cat((single_label[single_index], double_label[double_index]), dim=0)
                epoch_loss += -spearman_corr(pred, label).item()
            else:
                epoch_loss += -spearman_corr(single_pred[single_index], single_label[single_index]).item()
    return epoch_loss / len(loader)


def test_model(model, loader):

    model.eval()
    device = next(model.parameters()).device
    single_corr_dict = {}
    double_corr_dict = {}
    all_corr_dict = {}
    with torch.no_grad():
        for data, single_label, double_label, name in loader:
            data, single_label, double_label = to_gpu(data, device), to_gpu(single_label, device), to_gpu(double_label, device)
            single_pred, double_pred = model(data)
            single_index = torch.where(~torch.isnan(single_label))
            double_index = torch.where(~torch.isnan(double_label))
            single_corr_dict[name[0]] = spearman_corr(single_pred[single_index], single_label[single_index]).item()

            single_index = torch.where(~torch.isnan(single_label))
            double_index = torch.where(~torch.isnan(double_label))
            if len(double_index[0]) != 0:
                pred = torch.cat((single_pred[single_index], double_pred[double_index]), dim=0)
                label = torch.cat((single_label[single_index], double_label[double_index]), dim=0)
                double_corr_dict[name[0]] = spearman_corr(double_pred[double_index], double_label[double_index]).item()
                all_corr_dict[name[0]] = spearman_corr(pred, label).item()
            else:
                all_corr_dict[name[0]] = spearman_corr(single_pred[single_index], single_label[single_index]).item()
    return single_corr_dict, double_corr_dict, all_corr_dict
