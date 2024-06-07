import torch
from loss import spearman_loss
from metrics import spearman_corr


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
        return data, data["label"], data[self.dataset_type], name


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


def train_model(model, optimizer, loader, group_number):

    model.train()
    device = next(model.parameters()).device
    epoch_loss = 0
    epoch_corr = 0
    for data, label, select_index, _ in loader:
        data, label = to_gpu(data, device), to_gpu(label, device)

        optimizer.zero_grad()
        pred = model(data)

        # split big data into small groups to calculate spearman loss
        counts = 1
        if group_number:
            for i in range(0, len(select_index[0]), group_number):
                tmp_select_index = [select_index[0][i : i + group_number], select_index[1][i : i + group_number]]
                if len(tmp_select_index[0]) >= 16:
                    if i == 0:
                        loss = spearman_loss(pred[0][tmp_select_index].unsqueeze(0), label[0][tmp_select_index].unsqueeze(0), 1e-2, "kl")
                    else:
                        counts += 1
                        loss += spearman_loss(pred[0][tmp_select_index].unsqueeze(0), label[0][tmp_select_index].unsqueeze(0), 1e-2, "kl")
            loss /= counts
        else:
            loss = spearman_loss(pred[0][select_index].unsqueeze(0), label[0][select_index].unsqueeze(0), 1e-2, "kl")
        corr = spearman_corr(pred[0][select_index], label[0][select_index])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type=2, max_norm=10, error_if_nonfinite=True)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_corr += corr.item()
    return epoch_loss / len(loader), epoch_corr / len(loader)


def validation_model(model, loader):

    model.eval()
    device = next(model.parameters()).device
    epoch_loss = 0
    with torch.no_grad():
        for data, label, select_index, _ in loader:
            data, label = to_gpu(data, device), to_gpu(label, device)

            pred = model(data)
            loss = -spearman_corr(pred[0][select_index], label[0][select_index])
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def test_model(model, loader):

    model.eval()
    device = next(model.parameters()).device
    corr_sum = 0
    corr_dict = {}
    with torch.no_grad():
        for data, label, select_index, name in loader:
            data, label = to_gpu(data, device), to_gpu(label, device)

            pred = model(data)
            corr = spearman_corr(pred[0][select_index], label[0][select_index]).item()
            corr_sum += corr
            corr_dict[name[0]] = corr
    return corr_sum / len(loader), corr_dict
