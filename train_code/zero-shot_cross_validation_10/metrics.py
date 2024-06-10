import torch


def _find_repeats(data):

    temp = data.detach().clone()
    temp = temp.sort()[0]

    change = torch.cat([torch.tensor([True], device=temp.device), temp[1:] != temp[:-1]])
    unique = temp[change]
    change_idx = torch.cat([torch.nonzero(change), torch.tensor([[temp.numel()]], device=temp.device)]).flatten()
    freq = change_idx[1:] - change_idx[:-1]
    atleast2 = freq > 1
    return unique[atleast2]


def _rank_data(data):

    n = data.numel()
    rank = torch.empty_like(data)
    idx = data.argsort()
    rank[idx[:n]] = torch.arange(1, n + 1, dtype=data.dtype, device=data.device)

    repeats = _find_repeats(data)
    for r in repeats:
        condition = data == r
        rank[condition] = rank[condition].mean()
    return rank


def spearman_corr(pred, true):

    assert pred.dtype == true.dtype
    assert pred.ndim <= 2 and true.ndim <= 2

    if pred.ndim == 1:
        pred = _rank_data(pred)
        true = _rank_data(true)
    else:
        pred = torch.stack([_rank_data(p) for p in pred.T]).T
        true = torch.stack([_rank_data(t) for t in true.T]).T

    preds_diff = pred - pred.mean(0)
    target_diff = true - true.mean(0)

    cov = (preds_diff * target_diff).mean(0)
    preds_std = torch.sqrt((preds_diff * preds_diff).mean(0))
    target_std = torch.sqrt((target_diff * target_diff).mean(0))

    spearman_corr = cov / (preds_std * target_std + 1e-6)
    return torch.clamp(spearman_corr, -1.0, 1.0)
