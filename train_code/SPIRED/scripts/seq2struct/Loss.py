import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = 1e-8


def tmscore_torch(coords1, true_coords, dcut=15.0):
    N, L, _ = coords1.shape
    coords2 = torch.tile(true_coords[None, :, :], (N, 1, 1))

    center1 = torch.mean(coords1, dim=1, keepdim=True)
    center2 = torch.mean(coords2, dim=1, keepdim=True)

    coords1 -= center1
    coords2 -= center2

    cov_matrix = torch.bmm(coords1.permute(0, 2, 1), coords2)  # (L,3,3)
    u, _, v = torch.linalg.svd(cov_matrix)
    rotation = torch.bmm(u, v)

    coords1_rotated = torch.bmm(coords1, rotation)

    d = torch.linalg.norm(coords1_rotated - coords2, dim=-1)
    rmsd = torch.mean(d, dim=-1)
    d0 = 1.24 * (L - dcut) ** (1.0 / 3.0) - 1.8
    score = torch.sum(1 / (1 + (d / d0) ** 2), dim=-1) / L
    return score, rmsd


def tmscore_numpy(coords1, coords2, dcut=15.0):
    # coords1 = (L,3) CA coords
    L, _ = coords1.shape
    center1 = np.mean(coords1, axis=0)
    center2 = np.mean(coords2, axis=0)

    coords1 -= center1
    coords2 -= center2

    cov_matrix = np.dot(np.transpose(coords1), coords2)
    u, _, v = np.linalg.svd(cov_matrix)
    rotation = np.dot(u, v)

    coords1_rotated = np.dot(coords1, rotation)

    d = np.linalg.norm(coords1_rotated - coords2, axis=1)
    rmsd = np.mean(d, axis=-1)
    d0 = 1.24 * (L - dcut) ** (1.0 / 3.0) - 1.8
    score = np.sum(1 / (1 + (d / d0) ** 2)) / L
    return score, rmsd


def getLDDT(predcadist, truecadist):
    ### predcadist: (N,L,L,L)
    ### truecadist: (N,L,L)
    ###

    """

    jupyter notebook: /export/disk4/xyg/mixnet/analysis.ipynb

    D: true distance
    d: predicted distance

    s: minimum sequence separation. lDDT original paper: default s=0
    t: threshold [0.5,1,2,4] the same ones used to compute the GDT-HA score
    Dmax: inclusion radius,far definition,according to lDDT paper

    Referenece
    0. AlphaFold1 SI
    1. lDDT original paper: doi:10.1093/bioinformatics/btt473

    """

    N, B, L, L = predcadist.shape
    truecadist = torch.tile(truecadist[:, None, :, :], (1, B, 1, 1))
    predcadist = predcadist.to(truecadist.device)

    Dmax = 15.0
    maskfar = torch.as_tensor(truecadist <= Dmax, dtype=torch.float32)  # (N,L,L,L)

    s = 0  #  lDDT original paper: default s=0
    a = torch.arange(L).reshape([1, L]).to(maskfar.device)
    maskLocal = torch.as_tensor(torch.abs(a - a.T) >= s, dtype=torch.float32)  # (L,L)
    maskLocal = torch.tile(maskLocal[None, None, :, :], (N, B, 1, 1))
    fenmu = maskLocal * maskfar

    Ratio = 0
    t = [0.5, 1, 2, 4]  # the same ones used to compute the GDT-HA score
    for t0 in t:
        preserved = torch.as_tensor(torch.abs(truecadist - predcadist) < t0, dtype=torch.float32)
        fenzi = maskLocal * maskfar * preserved
        Ratio += torch.sum(fenzi, dim=3) / (torch.sum(fenmu, dim=3) + eps)
    lddt = Ratio / 4.0  # (N,L,L)  range (0,1]
    return lddt


def pLDDTloss(plddt, truelddt):

    plddt = plddt.to(truelddt.device)

    return torch.mean(torch.abs(plddt - truelddt))


def lossViolation(predcadist):
    ### non-bonded CA pair and peptide bond length
    ### ref: AF2 SI P40
    ### predcadist.shape (N,L,L,L)
    tau = 0.02  ## tolerance
    B, N, L, L = predcadist.shape

    a = torch.arange(L).reshape(1, L).to(predcadist.device)
    masklocal = torch.as_tensor(torch.abs(a - a.T) >= 2, dtype=torch.float32)  ## (L,L)
    keeplocal = 1 - masklocal - torch.eye(L).to(predcadist.device)

    masklocal = torch.tile(masklocal[None, None, :, :], (B, N, 1, 1))  # (N,L,L,L)
    keeplocal = torch.tile(keeplocal[None, None, :, :], (B, N, 1, 1))  # (N,L,L,L)

    Nclash = torch.sum(torch.as_tensor(3.80 - tau - predcadist * masklocal > 0, dtype=torch.float32)) + eps
    lossclash = torch.sum(F.relu(3.80 - tau - predcadist * masklocal, 0)) / Nclash  ## too short penalized
    lossbond = torch.mean(F.relu(torch.abs(predcadist * keeplocal - 3.82) - tau, 0))

    return lossclash, lossbond


def getRDloss_byResidue(diff, coord_idx):

    eps = 1e-8
    L = diff.shape[-1]
    if L > 256:
        diff = diff[:, :, coord_idx, ...]  # (B, 3, L/2, L)

    diff = diff[:, :, :, None, :] - diff[:, :, :, :, None]  # (N,3,L,L,L)
    diff = torch.sum(diff * diff, dim=1)  # (N,L,L,L)
    diff = torch.sqrt(diff + eps)  # (N,L,L,L)
    realFape = torch.mean(diff, dim=(2, 3))  # (N,L)

    return realFape[0]


def getRDloss(diff, mask, coord_idx):

    eps = 1e-8
    L = diff.shape[-1]
    if L > 256:
        diff = diff[:, :, coord_idx, :]  # (B, 3, L/2, L)

    diff = diff[:, :, :, None, :] - diff[:, :, :, :, None]  # (B, 3, L/2, L, L)
    diff = torch.sum(diff * diff, dim=1)  # (B, L/2, L, L)
    diff = torch.sqrt(diff + eps)  # (B, L/2, L, L)
    realFape = torch.mean(diff)

    N = diff.shape[1]
    mask = mask.to(diff.device)
    mask = torch.tile(mask[:, None, :, :], (1, N, 1, 1))  # (N,L,L,L)
    lossRD = torch.sum(diff * mask) / (torch.sum(mask) + eps)  # mean for each residue, including N proteins

    return lossRD, realFape


def getFapeLoss(diff, coord_idx, dclamp, ratio=0.1, lossformat="dclamp"):

    eps = 1e-8
    L = diff.shape[-2]
    if L > 256:
        diff = diff[:, :, coord_idx, ...]  # (B, 3, L/2, L)

    diff = diff[:, :, :, None, :] - diff[:, :, :, :, None]  # (B,3,L,L,L)
    diff = torch.sum(diff * diff, dim=1)  # (B,L,L,L)
    diff = torch.sqrt(diff + eps)
    realFape = torch.mean(diff)

    quantile = torch.tensor(0.0)

    if lossformat == "dclamp":
        mask = torch.as_tensor(diff <= dclamp, dtype=torch.float32)  # (N,L,L,L)
        mask = mask * (torch.ones([L, L]) - torch.eye(L)).to(diff.device)
        fapeLoss = torch.sum(diff * mask) / (torch.sum(mask) + eps)
    elif lossformat == "ReluDclamp":
        diff = ratio * nn.ReLU()(diff - dclamp) + dclamp - nn.ReLU()(dclamp - diff)
        mask = torch.ones_like(diff) * (torch.ones([L, L]) - torch.eye(L)).to(diff.device)
        fapeLoss = torch.sum(diff * mask) / (torch.sum(mask) + eps)
    elif lossformat == "NoDclamp":
        mask = torch.ones_like(diff) * (torch.ones([L, L]) - torch.eye(L)).to(diff.device)
        fapeLoss = torch.sum(diff * mask) / (torch.sum(mask) + eps)
    elif lossformat == "probDclamp":
        maskdclamp = torch.as_tensor(diff <= dclamp, dtype=torch.float32)  # (N,L,L,L)
        maskdclamp = maskdclamp * (torch.ones([L, L]) - torch.eye(L)).to(diff.device)
        maskprob = torch.as_tensor(torch.rand([L, L]) >= (1 - ratio), dtype=torch.float32)
        maskprob = torch.triu(maskprob, diagonal=1)
        maskprob = maskprob + maskprob.permute(1, 0)
        maskprob = maskprob.to(diff.device)
        mask = maskprob * (1 - maskdclamp) + maskdclamp
        fapeLoss = torch.sum(diff * mask) / (torch.sum(mask) + eps)

    del mask

    return fapeLoss, realFape, quantile


def getCadistLoss(cadistavg, labelcadist, maskNLL, option="RDloss"):

    cadistavg = cadistavg.to(labelcadist.device)

    if option == "RDloss":
        cadistDiff = torch.abs(labelcadist - cadistavg)  # (N,L,L)
        realcadistL1loss = torch.mean(cadistDiff)
        cadistL1loss = torch.sum(cadistDiff * maskNLL) / (torch.sum(maskNLL) + eps)
    elif option == "fapeLoss":
        cadistDiff = torch.abs(labelcadist - cadistavg)  # (N,L,L)
        realcadistL1loss = torch.mean(cadistDiff)
        cadistL1loss = torch.sum(cadistDiff * maskNLL) / (torch.sum(maskNLL) + eps)
    return cadistL1loss, realcadistL1loss


def getMaskRD(labelcadist, cutoff):

    mask = F.sigmoid(cutoff - labelcadist)
    L = labelcadist.shape[-2]
    mask = mask * (torch.ones((L, L)) - torch.eye(L)).to(labelcadist.device)
    return mask


def getMaskFape(cadistavg, labelcadist, cutoff):
    cadistavg = cadistavg.to(labelcadist.device)
    cadistDiffNLL = torch.abs(labelcadist - cadistavg)
    mask = torch.as_tensor(cadistDiffNLL <= cutoff, dtype=torch.float32)
    L = cadistavg.shape[-2]
    mask = mask * (torch.ones((L, L)) - torch.eye(L)).to(labelcadist.device)
    mask = mask
    return mask


def getLossCE(cb, omega, theta, phi, labelbatch):
    cb, omega, theta, phi = (cb.to(labelbatch.device), omega.to(labelbatch.device), theta.to(labelbatch.device), phi.to(labelbatch.device))
    labelcb = labelbatch[:, 1, :, :].long()
    labelomega = labelbatch[:, 2, :, :].long()
    labeltheta = labelbatch[:, 3, :, :].long()
    labelphi = labelbatch[:, 4, :, :].long()

    cbloss = F.cross_entropy(input=cb, target=labelcb)
    omegaloss = F.cross_entropy(input=omega, target=labelomega)
    thetaloss = F.cross_entropy(input=theta, target=labeltheta)
    philoss = F.cross_entropy(input=phi, target=labelphi)
    lossoptim = cbloss + omegaloss + thetaloss + philoss

    loss = {}
    loss["optim"] = lossoptim
    loss["cb"] = cbloss
    loss["omega"] = omegaloss
    loss["theta"] = thetaloss
    loss["phi"] = philoss
    return loss


def getLossCAL1(ca, labelbatch, cutoff):
    ca = ca.to(labelbatch.device)
    labelcadist = labelbatch[:, 0, :, :].long()  # (N,L,L)
    mask = torch.as_tensor(labelcadist <= cutoff, dtype=torch.float32)  # (N,L,L)
    return torch.sum(torch.abs(labelcadist - ca[:, 0, :, :]) * mask) / (torch.sum(mask) + eps)


def getLoss_phi_psi_1D_L1(phi_psi_1D, label):
    loss_L1 = nn.L1Loss()
    phi_psi_1D = phi_psi_1D.to(label.device)
    loss_phi_psi = loss_L1(phi_psi_1D, label)
    # print("loss_phi_psi", loss_phi_psi.item())
    return loss_phi_psi


def getLossBlockV0(Predxyz, PredCadistavg, Plddt, labelbatch, coord_idx):
    LossRD = {}
    LossCadist = {}
    LossRealfape = {}
    LossRealcadist = {}
    LossTruelddt = {}
    LossPlddt = {}
    Plddt_dict = {}
    LossClash = {}
    LossBond = {}

    Predxyz_0, Predxyz_1 = (Predxyz[0].to(labelbatch.device), Predxyz[1].to(labelbatch.device))
    labelcadist = labelbatch[:, 0, :, :]  # (L,L)
    truexyz = labelbatch[:, -3:, :, :]

    diff = Predxyz_0 - truexyz
    maskNLL = getMaskRD(labelcadist, cutoff=24.0)
    lossRD, realFape = getRDloss(diff, maskNLL, coord_idx)
    cadistloss, realcadistloss = getCadistLoss(PredCadistavg[0], labelcadist, maskNLL, option="RDloss")

    LossRD["RDloss24_1"] = lossRD
    LossCadist["RDloss24_1"] = cadistloss
    LossRealcadist["RDloss24_1"] = realcadistloss
    LossRealfape["RDloss24_1"] = realFape
    Plddt_dict["RDloss24_1"] = Plddt[0]

    #### RDloss24 and L1 loss
    diff = Predxyz_1 - truexyz
    maskNLL = getMaskRD(labelcadist, cutoff=24.0)
    lossRD, realFape = getRDloss(diff, maskNLL, coord_idx)
    cadistloss, realcadistloss = getCadistLoss(PredCadistavg[1], labelcadist, maskNLL, option="RDloss")

    # true lDDT and plDDT
    L = truexyz.shape[-1]
    if L > 256:
        xyzLLL3 = Predxyz_1[:, :, coord_idx, None, :] - Predxyz_1[:, :, coord_idx, :, None]  # (N,3,L/2,L,L)
        plddt_layer1 = Plddt[1][:, coord_idx, :]
    else:
        xyzLLL3 = Predxyz_1[:, :, :, None, :] - Predxyz_1[:, :, :, :, None]  # (N,3,L/2,L,L)
        plddt_layer1 = Plddt[1]

    predcadist = torch.sqrt(torch.sum(xyzLLL3 * xyzLLL3, dim=1) + eps)  # (N,L,L,L)
    truelddt = getLDDT(predcadist.detach(), labelcadist)
    plddtloss = pLDDTloss(plddt_layer1, truelddt)

    # clash loss
    LossClash["RDloss24_2"], LossBond["RDloss24_2"] = lossViolation(predcadist)

    LossRD["RDloss24_2"] = lossRD
    LossCadist["RDloss24_2"] = cadistloss
    LossRealcadist["RDloss24_2"] = realcadistloss
    LossRealfape["RDloss24_2"] = realFape
    LossTruelddt["RDloss24_2"] = truelddt
    LossPlddt["RDloss24_2"] = plddtloss
    Plddt_dict["RDloss24_2"] = Plddt[1]

    loss_RD = (LossRD["RDloss24_1"] + 0.1 * LossCadist["RDloss24_1"]) / 24 + (LossRD["RDloss24_2"] + 0.1 * LossCadist["RDloss24_2"]) / 24

    loss_plddt = LossPlddt["RDloss24_2"]

    loss_clash = (LossClash["RDloss24_2"] + LossBond["RDloss24_2"]) / 40

    lossoptim = (loss_RD / 2) + loss_plddt + loss_clash

    # print("\n", "loss_RD", loss_RD.item(), "loss_plddt", loss_plddt.item(), "loss_clash", loss_clash.item())
    # print("\n","LossClash_RDloss24_2", LossClash['RDloss24_2'].item(), "LossBond_RDloss24_2", LossBond['RDloss24_2'].item(), "\n")

    return lossoptim, LossRD, LossCadist, LossRealfape, LossRealcadist, LossPlddt, Plddt_dict, LossTruelddt


def getLossBlockV1(Predxyz, PredCadistavg, Plddt, labelbatch, coord_idx):

    LossRD = {}
    LossCadist = {}
    LossFape = {}
    LossRealfape = {}
    LossRealcadist = {}
    Fapequantile = {}
    LossTruelddt = {}
    LossPlddt = {}
    Plddt_dict = {}
    LossClash = {}
    LossBond = {}

    Predxyz_0, Predxyz_1, Predxyz_2, Predxyz_3, Predxyz_4, Predxyz_5 = (Predxyz[0].to(labelbatch.device), Predxyz[1].to(labelbatch.device), Predxyz[2].to(labelbatch.device), Predxyz[3].to(labelbatch.device), Predxyz[4].to(labelbatch.device), Predxyz[5].to(labelbatch.device))

    labelcadist = labelbatch[:, 0, :, :]  # (L,L)
    truexyz = labelbatch[:, -3:, :, :]

    diff = Predxyz_0 - truexyz
    maskNLL = getMaskRD(labelcadist, cutoff=32.0)
    lossRD, realFape = getRDloss(diff, maskNLL, coord_idx)
    cadistloss, realcadistloss = getCadistLoss(PredCadistavg[0], labelcadist, maskNLL, option="RDloss")

    LossRD["RDloss32_1"] = lossRD
    LossCadist["RDloss32_1"] = cadistloss
    LossRealcadist["RDloss32_1"] = realcadistloss
    LossRealfape["RDloss32_1"] = realFape
    Plddt_dict["RDloss32_1"] = Plddt[0]

    #### Layer2
    diff = Predxyz_1 - truexyz
    maskNLL = getMaskRD(labelcadist, cutoff=32.0)

    lossRD, realFape = getRDloss(diff, maskNLL, coord_idx)
    cadistloss, realcadistloss = getCadistLoss(PredCadistavg[1], labelcadist, maskNLL, option="RDloss")

    LossRD["RDloss32_2"] = lossRD
    LossCadist["RDloss32_2"] = cadistloss
    LossRealcadist["RDloss32_2"] = realcadistloss
    LossRealfape["RDloss32_2"] = realFape
    Plddt_dict["RDloss32_2"] = Plddt[1]

    #### Layer3
    diff = Predxyz_2 - truexyz
    maskNLL = getMaskRD(labelcadist, cutoff=32.0)

    lossRD, realFape = getRDloss(diff, maskNLL, coord_idx)
    cadistloss, realcadistloss = getCadistLoss(PredCadistavg[2], labelcadist, maskNLL, option="RDloss")

    LossRD["RDloss32_3"] = lossRD
    LossCadist["RDloss32_3"] = cadistloss
    LossRealcadist["RDloss32_3"] = realcadistloss
    LossRealfape["RDloss32_3"] = realFape
    Plddt_dict["RDloss32_3"] = Plddt[2]

    #### Layer4
    diff = Predxyz_3 - truexyz
    maskNLL = getMaskRD(labelcadist, cutoff=32.0)

    lossRD, realFape = getRDloss(diff, maskNLL, coord_idx)
    cadistloss, realcadistloss = getCadistLoss(PredCadistavg[3], labelcadist, maskNLL, option="RDloss")

    LossRD["RDloss32_4"] = lossRD
    LossCadist["RDloss32_4"] = cadistloss
    LossRealcadist["RDloss32_4"] = realcadistloss
    LossRealfape["RDloss32_4"] = realFape
    Plddt_dict["RDloss32_4"] = Plddt[3]

    #### Layer5
    diff = Predxyz_4 - truexyz

    fapeLoss, realFape, fapequantile = getFapeLoss(diff, coord_idx, dclamp=40.0, ratio=0.1, lossformat="probDclamp")

    maskNLL = getMaskFape(PredCadistavg[4], labelcadist, cutoff=40.0)
    cadistloss, realcadistloss = getCadistLoss(PredCadistavg[4], labelcadist, maskNLL, option="fapeLoss")
    LossFape["fape40_1"] = fapeLoss
    LossRealfape["fape40_1"] = realFape
    LossCadist["fape40_1"] = cadistloss
    LossRealcadist["fape40_1"] = realcadistloss
    Fapequantile["fape40_1"] = fapequantile
    Plddt_dict["fape40_1"] = Plddt[4]

    #### Layer6
    diff = Predxyz_5 - truexyz
    lossRD_byResidue = getRDloss_byResidue(diff, coord_idx)

    fapeLoss, realFape, fapequantile = getFapeLoss(diff, coord_idx, dclamp=40.0, ratio=0.1, lossformat="probDclamp")

    maskNLL = getMaskFape(PredCadistavg[5], labelcadist, cutoff=40)
    cadistloss, realcadistloss = getCadistLoss(PredCadistavg[5], labelcadist, maskNLL, option="fapeLoss")

    # true lDDT and plDDT
    L = truexyz.shape[-1]
    if L > 256:
        xyzLLL3 = Predxyz_5[:, :, coord_idx, None, :] - Predxyz_5[:, :, coord_idx, :, None]  # (N,3,L,L,L)
        plddt_layer5 = Plddt[5][:, coord_idx, :]
    else:
        xyzLLL3 = Predxyz_5[:, :, :, None, :] - Predxyz_5[:, :, :, :, None]  # (N,3,L,L,L)
        plddt_layer5 = Plddt[5]

    predcadist = torch.sqrt(torch.sum(xyzLLL3 * xyzLLL3, dim=1) + eps)  # (N,L,L,L)
    truelddt = getLDDT(predcadist.detach(), labelcadist)
    plddtloss = pLDDTloss(plddt_layer5, truelddt)

    # clash Loss
    LossClash["fape40_2"], LossBond["fape40_2"] = lossViolation(predcadist)

    LossFape["fape40_2"] = fapeLoss
    LossRealfape["fape40_2"] = realFape
    LossCadist["fape40_2"] = cadistloss
    LossRealcadist["fape40_2"] = realcadistloss
    Fapequantile["fape40_2"] = fapequantile
    LossPlddt["fape40_2"] = plddtloss
    LossTruelddt["fape40_2"] = truelddt
    Plddt_dict["fape40_2"] = Plddt[5]

    Loss_RD = (LossRD["RDloss32_1"] + 0.1 * LossCadist["RDloss32_1"]) / 32 + (LossRD["RDloss32_2"] + 0.1 * LossCadist["RDloss32_2"]) / 32 + (LossRD["RDloss32_3"] + 0.1 * LossCadist["RDloss32_3"]) / 32 + (LossRD["RDloss32_4"] + 0.1 * LossCadist["RDloss32_4"]) / 32

    Loss_Fape = (LossFape["fape40_1"] + 0.1 * LossCadist["fape40_1"]) / 40 + (LossFape["fape40_2"] + 0.1 * LossCadist["fape40_2"]) / 40

    Loss_clash = (LossClash["fape40_2"] + LossBond["fape40_2"]) / 40

    Loss_plddt = LossPlddt["fape40_2"]

    # print("\n", "Loss_RD", Loss_RD.item(), "Loss_Fape", Loss_Fape.item(), "Loss_clash", Loss_clash.item(), "Loss_plddt", Loss_plddt.item())

    lossoptim = (Loss_RD + Loss_Fape) / 6 + Loss_plddt + Loss_clash

    return lossoptim, LossRD, LossFape, LossCadist, LossRealfape, LossRealcadist, LossPlddt, Plddt_dict, LossTruelddt, lossRD_byResidue


def getLossMultBlock(Predxyz, PredCadistavg, Plddt, labelbatch, coord_idx):

    LossRD = {}
    LossFape = {}
    LossCadist = {}
    LossRealfape = {}
    LossRealcadist = {}
    LossTruelddt = {}
    LossPlddt = {}
    Plddt_dict = {}

    Blocks = list(Predxyz.keys())
    block = Blocks[0]
    lossoptim, LossRD0, LossCadist0, LossRealfape0, LossRealcadist0, LossPlddt0, Plddt_dict0, LossTruelddt0 = getLossBlockV0(Predxyz[block], PredCadistavg[block], Plddt[block], labelbatch, coord_idx)
    LossRD[block] = LossRD0
    LossCadist[block] = LossCadist0
    LossRealfape[block] = LossRealfape0
    LossRealcadist[block] = LossRealcadist0
    LossTruelddt[block] = LossTruelddt0
    LossPlddt[block] = LossPlddt0
    Plddt_dict[block] = Plddt_dict0

    lossoptim = lossoptim.to(labelbatch.device)
    for block in Blocks[1:3]:
        tmp, LossRD0, LossCadist0, LossRealfape0, LossRealcadist0, LossPlddt0, Plddt_dict0, LossTruelddt0 = getLossBlockV0(Predxyz[block], PredCadistavg[block], Plddt[block], labelbatch, coord_idx)
        LossRD[block] = LossRD0
        LossCadist[block] = LossCadist0
        LossRealfape[block] = LossRealfape0
        LossRealcadist[block] = LossRealcadist0
        LossTruelddt[block] = LossTruelddt0
        LossPlddt[block] = LossPlddt0
        Plddt_dict[block] = Plddt_dict0
        # add loss
        lossoptim += tmp.to(labelbatch.device)

    block = Blocks[3]
    tmp, LossRD0, LossFape0, LossCadist0, LossRealfape0, LossRealcadist0, LossPlddt0, Plddt_dict0, LossTruelddt0, lossRD_byResidue = getLossBlockV1(Predxyz[block], PredCadistavg[block], Plddt[block], labelbatch, coord_idx)
    LossRD[block] = LossRD0
    LossFape[block] = LossFape0
    LossCadist[block] = LossCadist0
    LossRealfape[block] = LossRealfape0
    LossRealcadist[block] = LossRealcadist0
    LossTruelddt[block] = LossTruelddt0
    LossPlddt[block] = LossPlddt0
    Plddt_dict[block] = Plddt_dict0

    lossoptim += tmp.to(labelbatch.device)

    lossoptim = lossoptim / 4

    loss = {}
    loss["optim"] = lossoptim
    loss["RD"] = LossRD
    loss["Fape"] = LossFape
    loss["Cadist"] = LossCadist
    loss["RealFape"] = LossRealfape
    loss["RealCadist"] = LossRealcadist
    loss["plddt_loss"] = LossPlddt
    loss["plddt"] = Plddt_dict
    loss["truelddt"] = LossTruelddt
    return loss, lossRD_byResidue
