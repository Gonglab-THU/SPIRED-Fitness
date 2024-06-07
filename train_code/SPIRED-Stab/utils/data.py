import torch
import pandas as pd
from loss import spearman_loss
from metrics import spearman_corr

def convert_batch_data(data, batch_size, seed):
    
    for name in data.index.unique():
        remainder = len(data.loc[name]) % batch_size
        if remainder != 0:
            data = pd.concat((data, data.loc[name].sample(batch_size - remainder, random_state = seed)))
    data = data.groupby(data.index).apply(lambda x: x.sample(frac = 1, random_state = seed)).droplevel(0)
    data['group'] = [str(second) for first in [[i] * batch_size for i in range(len(data) // batch_size)] for second in first]
    data = data.reset_index().set_index(['group'])
    
    data = data.loc[pd.Series([str(i) for i in range(len(data) // batch_size)]).sample(frac = 1, random_state = seed).tolist()]
    data.reset_index(drop = True, inplace = True)
    return data

class ProcessingData(torch.utils.data.Dataset):
    
    def __init__(self, data_csv, wt_data_features):
        
        self.data_csv = data_csv
        self.wt_data_features = wt_data_features
    
    def __len__(self):
        
        return len(self.data_csv)
    
    def __getitem__(self, index):
        
        name = self.data_csv.loc[index, 'name']
        mut_info = self.data_csv.loc[index, 'mut_info']
        mut_pos_tmp = self.data_csv.loc[index, 'mut_pos']
        label = self.data_csv.loc[index, 'label']
        
        mut_data = {}
        mut_data['embedding'] = torch.load(f'/data/embedding/{name}/{mut_info}.pt')
        mut_data['plddt'] = torch.load(f'/data/plddt/{name}/{mut_info}.pt')
        mut_data['pair'] = torch.load(f'/data/pair/{name}/{mut_info}.pt')
        
        mut_pos = [0] * mut_data['pair'].shape[0]
        for i in mut_pos_tmp:
            mut_pos[i] = 1
        
        return self.wt_data_features[name], mut_data, torch.tensor(mut_pos), label, name + '_' + mut_info

def to_gpu(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(device = device, non_blocking = True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [to_gpu(i, device = device) for i in obj]
    elif isinstance(obj, tuple):
        return (to_gpu(i, device = device) for i in obj)
    elif isinstance(obj, dict):
        return {i: to_gpu(j, device = device) for i, j in obj.items()}
    else:
        return obj

def train_model(model, optimizer, loader):
    
    model.train()
    device = next(model.parameters()).device
    epoch_loss = 0
    epoch_corr = 0
    for wt_data, mut_data, mut_pos, label, _ in loader:
        wt_data, mut_data, mut_pos, label = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(mut_pos, device), to_gpu(label, device)
        
        optimizer.zero_grad()
        pred = model(wt_data, mut_data, mut_pos)
        
        loss = spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), 1e-2, 'kl')
        corr = spearman_corr(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type = 2, max_norm = 10, error_if_nonfinite = True)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_corr += corr.item()
    return epoch_loss / len(loader), epoch_corr / len(loader)

def validation_model(model, loader):
    
    model.eval()
    device = next(model.parameters()).device
    result = pd.DataFrame()
    with torch.no_grad():
        for wt_data, mut_data, mut_pos, label, mut_name in loader:
            wt_data, mut_data, mut_pos, label = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(mut_pos, device), to_gpu(label, device)
            
            pred = model(wt_data, mut_data, mut_pos)
            tmp = pd.DataFrame(index = mut_name, data = {'pred': pred.detach().cpu().numpy(), 'label': label.detach().cpu().numpy()})
            result = pd.concat((result, tmp))
    result['protein_name'] = result.index.str.rsplit('_', n = 1).str[0]
    return -result.groupby('protein_name').corr('spearman').iloc[0::2, 1].mean()

def test_model(model, loader):
    
    model.eval()
    device = next(model.parameters()).device
    result = pd.DataFrame()
    with torch.no_grad():
        for wt_data, mut_data, mut_pos, label, mut_name in loader:
            wt_data, mut_data, mut_pos = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(mut_pos, device)
            
            pred = model(wt_data, mut_data, mut_pos)
            tmp = pd.DataFrame(index = mut_name, data = {'pred': pred.detach().cpu().numpy(), 'label': label.detach().cpu().numpy()})
            result = pd.concat((result, tmp))
    return result
