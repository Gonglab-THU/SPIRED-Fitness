import torch
import numpy as np
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
    data.set_index('group', inplace = True)
    
    data = data.loc[pd.Series([str(i) for i in range(len(data) // batch_size)]).sample(frac = 1, random_state = seed).tolist()]
    data.reset_index(drop = True, inplace = True)
    return data

class ProcessingData(torch.utils.data.Dataset):
    
    def __init__(self, data_csv):
        
        self.data_csv = data_csv
    
    def __len__(self):
        
        return len(self.data_csv)
    
    def __getitem__(self, index):
        
        mut_name = self.data_csv.loc[index, 'mut_name']
        wt_name = mut_name.rsplit('_', 1)[0]
        ddG_label = self.data_csv.loc[index, 'ddG'].astype(np.float32)
        dTm_label = self.data_csv.loc[index, 'dTm'].astype(np.float32)
        wt_seq = self.data_csv.loc[index, 'wt_seq']
        mut_seq = self.data_csv.loc[index, 'mut_seq']
        file = '/data'
        
        wt_data = {
            'embedding': torch.load(f'{file}/wt/{wt_name}/embedding.pt'),
            'plddt': torch.load(f'{file}/wt/{wt_name}/plddt.pt'),
            'pair': torch.load(f'{file}/wt/{wt_name}/pair.pt')
        }
        mut_data = {
            'embedding': torch.load(f'{file}/mut/{mut_name}/embedding.pt'),
            'plddt': torch.load(f'{file}/mut/{mut_name}/plddt.pt'),
            'pair': torch.load(f'{file}/mut/{mut_name}/pair.pt')
        }
        
        mut_pos = (np.array(list(wt_seq)) != np.array(list(mut_seq))).astype(int).tolist()
        return wt_data, mut_data, torch.tensor(mut_pos), ddG_label, dTm_label, mut_name

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
    length = 0
    for wt_data, mut_data, mut_pos, ddG_label, dTm_label, _ in loader:
        wt_data, mut_data, mut_pos, ddG_label, dTm_label = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(mut_pos, device), to_gpu(ddG_label, device), to_gpu(dTm_label, device)
        
        optimizer.zero_grad()
        ddG_pred, dTm_pred = model(wt_data, mut_data, mut_pos)
        
        if torch.isnan(ddG_label).sum().item() > 0 and torch.isnan(dTm_label).sum().item() > 0:
            continue
        elif torch.isnan(ddG_label).sum().item() > 0 and torch.isnan(dTm_label).sum().item() == 0:
            loss = spearman_loss(dTm_pred.unsqueeze(0), dTm_label.unsqueeze(0), 1e-2, 'kl')
            corr = spearman_corr(dTm_pred, dTm_label)
            epoch_loss += loss.item()
            epoch_corr += corr.item()
            length += 1
        elif torch.isnan(ddG_label).sum().item() == 0 and torch.isnan(dTm_label).sum().item() > 0:
            loss = spearman_loss(ddG_pred.unsqueeze(0), ddG_label.unsqueeze(0), 1e-2, 'kl')
            corr = spearman_corr(ddG_pred, ddG_label)
            epoch_loss += loss.item()
            epoch_corr += corr.item()
            length += 1
        else:
            loss = 0.5 * spearman_loss(ddG_pred.unsqueeze(0), ddG_label.unsqueeze(0), 1e-2, 'kl') + 0.5 * spearman_loss(dTm_pred.unsqueeze(0), dTm_label.unsqueeze(0), 1e-2, 'kl')
            corr = 0.5 * spearman_corr(ddG_pred, ddG_label) + 0.5 * spearman_corr(dTm_pred, dTm_label)
            epoch_loss += loss.item()
            epoch_corr += corr.item()
            length += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type = 2, max_norm = 10, error_if_nonfinite = True)
        optimizer.step()
    
    return epoch_loss / length, epoch_corr / length

def validation_model(model, loader):
    
    model.eval()
    device = next(model.parameters()).device
    result = pd.DataFrame()
    with torch.no_grad():
        for wt_data, mut_data, mut_pos, ddG_label, dTm_label, mut_name in loader:
            wt_data, mut_data, mut_pos = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(mut_pos, device)
            
            ddG_pred, dTm_pred = model(wt_data, mut_data, mut_pos)
            tmp = pd.DataFrame(index = mut_name, data = {'ddG_pred': ddG_pred.detach().cpu().numpy(), 'dTm_pred': dTm_pred.detach().cpu().numpy(), 'ddG_label': ddG_label.detach().cpu().numpy(), 'dTm_label': dTm_label.detach().cpu().numpy()})
            result = pd.concat((result, tmp))
    result['protein_name'] = result.index.str.rsplit('_', n = 1).str[0]
    return -result.groupby('protein_name').corr('spearman').iloc[2::4, 0].mean(), -result.groupby('protein_name').corr('spearman').iloc[3::4, 1].mean()

def test_model(model, loader, test_csv):
    
    model.eval()
    device = next(model.parameters()).device
    test_csv = test_csv.set_index('mut_name').copy()
    with torch.no_grad():
        for wt_data, mut_data, mut_pos, _, _, mut_name in loader:
            wt_data, mut_data, mut_pos = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(mut_pos, device)
            
            ddG_pred, dTm_pred = model(wt_data, mut_data, mut_pos)
            test_csv.loc[mut_name[0], 'ddG_pred'] = ddG_pred.item()
            test_csv.loc[mut_name[0], 'dTm_pred'] = dTm_pred.item()
    
    return test_csv

def train_mse_model(model, optimizer, loader, object):
    
    model.train()
    device = next(model.parameters()).device
    epoch_loss = 0
    length = 0
    if object == 'ddG':
        for wt_data, mut_data, mut_pos, ddG_label, dTm_label, _ in loader:
            wt_data, mut_data, mut_pos, ddG_label, dTm_label = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(mut_pos, device), to_gpu(ddG_label, device), to_gpu(dTm_label, device)
            
            optimizer.zero_grad()
            ddG_pred, dTm_pred = model(wt_data, mut_data, mut_pos)
            if torch.isnan(ddG_label).all().item():
                continue
            ddG_pred = ddG_pred[~torch.isnan(ddG_label)]
            ddG_label = ddG_label[~torch.isnan(ddG_label)]
            loss = torch.nn.functional.mse_loss(ddG_pred, ddG_label)
            epoch_loss += loss.item() * len(ddG_pred)
            length += len(ddG_pred)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type = 2, max_norm = 10, error_if_nonfinite = True)
            optimizer.step()
    elif object == 'dTm':
        for wt_data, mut_data, mut_pos, ddG_label, dTm_label, _ in loader:
            wt_data, mut_data, mut_pos, ddG_label, dTm_label = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(mut_pos, device), to_gpu(ddG_label, device), to_gpu(dTm_label, device)
            
            optimizer.zero_grad()
            ddG_pred, dTm_pred = model(wt_data, mut_data, mut_pos)
            if torch.isnan(dTm_label).all().item():
                continue
            dTm_pred = dTm_pred[~torch.isnan(dTm_label)]
            dTm_label = dTm_label[~torch.isnan(dTm_label)]
            loss = torch.nn.functional.mse_loss(dTm_pred, dTm_label)
            epoch_loss += loss.item() * len(dTm_pred)
            length += len(dTm_pred)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type = 2, max_norm = 10, error_if_nonfinite = True)
            optimizer.step()
    
    return epoch_loss / length

def validation_mse_model(model, loader):
    
    model.eval()
    device = next(model.parameters()).device
    result = pd.DataFrame()
    with torch.no_grad():
        for wt_data, mut_data, mut_pos, ddG_label, dTm_label, mut_name in loader:
            wt_data, mut_data, mut_pos = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(mut_pos, device)
            
            ddG_pred, dTm_pred = model(wt_data, mut_data, mut_pos)
            tmp = pd.DataFrame(index = mut_name, data = {'ddG_pred': ddG_pred.detach().cpu().numpy(), 'dTm_pred': dTm_pred.detach().cpu().numpy(), 'ddG_label': ddG_label.detach().cpu().numpy(), 'dTm_label': dTm_label.detach().cpu().numpy()})
            result = pd.concat((result, tmp))
    
    ddG = result[result['ddG_label'].notnull()].copy()
    ddG_mse_loss = (result['ddG_pred'] - result['ddG_label']).pow(2).mean()
    dTm = result[result['dTm_label'].notnull()].copy()
    dTm_mse_loss = (result['dTm_pred'] - result['dTm_label']).pow(2).mean()
    return ddG_mse_loss, dTm_mse_loss
