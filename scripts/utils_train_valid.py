import torch
import numpy as np
import time


def dist2bins(mat):
    """
    This function is for label
    bins definition: left open, right close. Because cutoff 8 definition: d<=8
    two kinds of bin length: 0.5,1.0
    bins:3.0],(3.0,3.5],(3.5,4.0],...,(21.5,22],(22,23],(23,24],...,(29,30],(30,
    Nbin=(22-3)*2+1+(30-22)+1=48
    mat: real value matrix, (L,L)
    return: (L,L,Nbins)

    """
    bins = list(np.arange(3.0, 22, 0.5)) + list(np.arange(22.0, 31.0, 1.0))
    L = mat.shape[0]
    Nbin = len(bins) + 1
    tensor = np.zeros([L, L, Nbin])
    tensor[..., 0] = (mat <= bins[0]).astype(int)
    for i in range(len(bins) - 1):
        tensor[..., i + 1] = ((mat > bins[i]) * (mat <= bins[i + 1])).astype(int)
    tensor[..., -1] = (mat > bins[-1]).astype(int)
    return tensor


def omega2bins(mat, distmat, dmax=30.0):
    """
    This function is for label
    bins definition: left open, right close
    anglebin=(beg0,end0]
    first bin=[-180,-165]
    Nbin=24+1  (1: non contact, i.e.out of max distance www.pnas.org/cgi/doi/10.1073/pnas.1914677117)
    because phi and theta is asymmetry, both two for loop begins from zero
    """
    L = mat.shape[0]
    bins = list(np.arange(-180, 180 + 15, 15))
    Nbin = (len(bins) - 1) + 1
    result = np.zeros([L, L, Nbin])
    result[..., 0] = (mat >= bins[0]).astype(float) * (mat <= bins[1]).astype(float)
    for i in range(1, Nbin - 1):
        result[..., i] = (mat > bins[i]).astype(float) * (mat <= bins[i + 1]).astype(float)

    contact = (distmat <= dmax).astype(float)
    contact = np.tile(contact[..., None], [1, 1, Nbin - 1])
    result[..., : Nbin - 1] = result[..., : Nbin - 1] * contact
    noncontact = (distmat > dmax).astype(float)
    masknan = (mat >= -180).astype(float)
    result[..., -1] = noncontact * masknan
    return result


def theta2bins(mat, distmat, dmax=30):
    return omega2bins(mat, distmat, dmax=30)


def phi2bins(mat, distmat, dmax=30):
    """
    This function is for label
    bins definition: left open, right close
    anglebin=(beg0,end0]
    first bin=[0,15]
    Nbin=12+1  (+non contact, i.e.out of max distance www.pnas.org/cgi/doi/10.1073/pnas.1914677117)
    because phi and theta is asymmetry, both two for loop begins from zero
    """
    L = mat.shape[0]
    bins = list(np.arange(0, 180 + 15, 15))
    Nbin = (len(bins) - 1) + 1
    result = np.zeros([L, L, Nbin])
    result[..., 0] = (mat >= bins[0]).astype(float) * (mat <= bins[1]).astype(float)
    for i in range(1, Nbin - 1):
        result[..., i] = (mat > bins[i]).astype(float) * (mat <= bins[i + 1]).astype(float)
    contact = (distmat <= dmax).astype(float)
    contact = np.tile(contact[..., None], [1, 1, Nbin - 1])
    result[..., : Nbin - 1] = result[..., : Nbin - 1] * contact
    noncontact = (distmat > dmax).astype(float)
    masknan = (mat >= 0).astype(float)
    result[..., -1] = noncontact * masknan
    return result


def cbLabelIndex(mat):
    cbBins = dist2bins(mat)
    L = mat.shape[0]
    binIndex = np.tile(np.arange(48)[None, None, :], [L, L, 1])
    cbIndex = np.array(np.sum(cbBins * binIndex, axis=2), dtype=int)
    return cbIndex


def omegaLabelIndex(mat, distmat):
    omegaBins = omega2bins(mat, distmat)
    L = mat.shape[0]
    binIndex = np.tile(np.arange(25)[None, None, :], [L, L, 1])
    omegaIndex = np.array(np.sum(omegaBins * binIndex, axis=2), dtype=int)
    return omegaIndex


def thetaLabelIndex(mat, distmat):
    return omegaLabelIndex(mat, distmat)


def phiLabelIndex(mat, distmat):
    phiBins = phi2bins(mat, distmat)
    L = mat.shape[0]
    binIndex = np.tile(np.arange(13)[None, None, :], [L, L, 1])
    phiIndex = np.array(np.sum(phiBins * binIndex, axis=2), dtype=int)
    return phiIndex


def get_dihedrals(a, b, c, d):
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:, None]

    v = b0 - np.sum(b0 * b1, axis=-1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=-1)[:, None] * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)

    return np.arctan2(y, x)


def get_angles(a, b, c):
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:, None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:, None]

    x = np.sum(v * w, axis=1)

    return np.arccos(x)


def get_info(xyz, dmax):
    nres = xyz.shape[0]
    N = xyz[:, 0, :]
    CA = xyz[:, 1, :]
    C = xyz[:, 2, :]
    CB = xyz[:, 3, :]
    d_CA_CB = CB - CA
    d_CA_CB = np.sqrt(np.sum(d_CA_CB * d_CA_CB, axis=-1))
    gly_index = np.where(d_CA_CB < 0.001)[0]  # for glycine, CB is overlapped with CA, d_CA_CB = 0

    # recreate Cb given N,Ca,C
    b = CA - N
    c = C - CA
    a = np.cross(b, c)
    # CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

    # fast neighbors search
    kdCb = scipy.spatial.cKDTree(CB)
    indices = kdCb.query_ball_tree(kdCb, dmax)
    # indices of contacting residues
    idx = np.array([[i, j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]

    # Cb-Cb distance matrix
    dist6d_CB = np.zeros((nres, nres))
    dist6d_CB[idx0, idx1] = np.linalg.norm(CB[idx1] - CB[idx0], axis=-1)

    # Ca-Ca distance matrix
    dist6d_CA = np.zeros((nres, nres))
    dist6d_CA[idx0, idx1] = np.linalg.norm(CA[idx1] - CA[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.full([nres, nres], np.nan)
    omega6d[idx0, idx1] = np.rad2deg(get_dihedrals(CA[idx0], CB[idx0], CB[idx1], CA[idx1]))
    omega6d[gly_index, :] = np.nan
    omega6d[:, gly_index] = np.nan

    # matrix of polar coord theta
    theta6d = np.full([nres, nres], np.nan)
    theta6d[idx0, idx1] = np.rad2deg(get_dihedrals(N[idx0], CA[idx0], CB[idx0], CB[idx1]))
    theta6d[gly_index, :] = np.nan
    theta6d[:, gly_index] = np.nan

    # matrix of polar coord phi
    phi6d = np.full([nres, nres], np.nan)
    phi6d[idx0, idx1] = np.rad2deg(get_angles(CA[idx0], CB[idx0], CB[idx1]))
    phi6d[gly_index, :] = np.nan
    phi6d[:, gly_index] = np.nan

    return (dist6d_CB, dist6d_CA, omega6d, theta6d, phi6d)


def NormVec(V):
    eps = 1e-10
    axis_x = V[:, 2] - V[:, 1]
    axis_x /= torch.norm(axis_x, dim=-1).unsqueeze(1) + eps
    axis_y = V[:, 0] - V[:, 1]
    axis_z = torch.cross(axis_x, axis_y, dim=1)
    axis_z /= torch.norm(axis_z, dim=-1).unsqueeze(1) + eps
    axis_y = torch.cross(axis_z, axis_x, dim=1)
    Vec = torch.stack([axis_x, axis_y, axis_z], dim=1)
    return Vec


def comp_feature(atoms):
    # atom: torch.tensor()  shape:[L, 4, 3] -> [L, [N, CA, C, CB], [x, y, z]]
    atoms_numpy = atoms
    atoms = torch.tensor(atoms)
    N_CA_C = atoms[:, :-1, :]  # [L, [N, CA, C, CB], [x, y, z]]
    rotation = NormVec(N_CA_C)  # N_CA_C, atoms=(L,3,3)
    r = torch.linalg.inv(rotation)

    xyz_CA = torch.einsum("a b i, a i j -> a b j", atoms[:, 1].unsqueeze(0) - atoms[:, 1].unsqueeze(1), r)  # (L,L,3)

    CB_dist, CA_dist, omega, theta, phi = get_info(atoms_numpy, dmax=10000)
    CA_dist = CA_dist[None]  # (1,L,L)
    cbIndex = cbLabelIndex(CB_dist)[None]
    omegaIndex = omegaLabelIndex(omega, CB_dist)[None]
    thetaIndex = thetaLabelIndex(theta, CB_dist)[None]
    phiIndex = phiLabelIndex(phi, CB_dist)[None]
    label_5 = np.concatenate([CA_dist, cbIndex, omegaIndex, thetaIndex, phiIndex], axis=0)
    label_5 = torch.from_numpy(label_5)

    label_all = torch.cat([label_5, xyz_CA.permute(2, 0, 1)], dim=0)
    return label_all  # (8,L,L)


def return_predcadist_truelddt(Predxyz, labelbatch):
    eps = 1e-8
    xyz_LL3 = Predxyz["4th"][-1]  # (batch=1,3,L,L)
    batch, _, L, _ = xyz_LL3.shape
    xyz_LL3 = xyz_LL3.to(labelbatch.device)

    xyz_LLL3 = xyz_LL3[:, :, :, None, :] - xyz_LL3[:, :, :, :, None]  # (N,3,L,L,L)
    predcadist = torch.sqrt(torch.sum(xyz_LLL3 * xyz_LLL3, dim=1) + eps)  # (N,L,L,L)

    labelcadist = labelbatch[:, 0, :, :]
    truelddt = Loss.getLDDT(predcadist, labelcadist)  # (batch,L,L)
    truelddt = torch.mean(torch.mean(truelddt, dim=0), dim=1)  # (L,)

    window = L // 18
    if window > 0:
        print("true_lddt window", window)
        truelddt_16 = list(torch.split(truelddt, window))[1:17]
        truelddt_16 = torch.stack(truelddt_16)  # (16, window)
        truelddt_16max = torch.max(truelddt_16, dim=-1)
        index_1 = truelddt_16max[1]
        truelddt_16max_index = index_1 + torch.arange(window, window * 17, window).to(device=index_1.device)
    else:
        truelddt_16max_index = torch.max(truelddt)[1]

    predcadist_best = torch.mean(predcadist[:, truelddt_16max_index, :, :], dim=1).unsqueeze(-1)

    bins = torch.cat([torch.arange(3.0, 22.0, 0.5), torch.arange(22.0, 31.0, 1.0)], dim=0).to(device=index_1.device)
    upper = torch.cat([bins[1:], bins.new_tensor([1e8])], dim=-1).to(device=index_1.device)
    predcadist_onehot = ((predcadist_best > bins) * (predcadist_best < upper)).type(xyz_LL3.dtype)  # (batch, L, L, 47)
    truelddt_max = torch.max(truelddt_16max[0]).item()

    return (predcadist_onehot, truelddt_max)


def return_predcadist_plddt(Predxyz, Plddt):
    eps = 1e-8
    xyz_LL3 = Predxyz["4th"][-1]  # (batch=1,3,L,L)
    batch, _, L, _ = xyz_LL3.shape

    xyz_LLL3 = xyz_LL3[:, :, :, None, :] - xyz_LL3[:, :, :, :, None]  # (N,3,L,L,L)
    predcadist = torch.sqrt(torch.sum(xyz_LLL3 * xyz_LLL3, dim=1) + eps)  # (N,L,L,L)

    plddt_tensor = Plddt["4th"][-1]  # plddt_tensor=(1,L,L)
    plddt_tensor = torch.mean(torch.mean(plddt_tensor, dim=0), dim=1)

    window = L // 18
    # print("true_lddt window", window)
    plddt_16 = list(torch.split(plddt_tensor, window))[1:17]
    plddt_16 = torch.stack(plddt_16)  # (16, window)
    plddt_16max = torch.max(plddt_16, dim=-1)
    index_1 = plddt_16max[1]
    plddt_16max_index = index_1 + torch.arange(window, window * 17, window).to(device=index_1.device)
    # print("plddt_16max_index", plddt_16max_index)

    predcadist_best = torch.mean(predcadist[:, plddt_16max_index, :, :], dim=1).unsqueeze(-1)

    bins = torch.cat([torch.arange(3.0, 22.0, 0.5), torch.arange(22.0, 31.0, 1.0)], dim=0).to(device=index_1.device)
    upper = torch.cat([bins[1:], bins.new_tensor([1e8])], dim=-1).to(device=index_1.device)
    predcadist_onehot = ((predcadist_best > bins) * (predcadist_best < upper)).type(xyz_LL3.dtype)  # (batch, L, L, 47)
    plddt_max = torch.max(plddt_16max[0]).item()

    return (predcadist_onehot, plddt_max)


def tokens2seq(token_array):
    # X: rare amino acid or unknown amino acid;  "-": gap;
    amino_acid_dict = {
        0: "A",
        1: "C",
        2: "D",
        3: "E",
        4: "F",
        5: "G",
        6: "H",
        7: "I",
        8: "K",
        9: "L",
        10: "M",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "T",
        17: "V",
        18: "W",
        19: "Y",
        20: "X",
        21: "O",
        22: "U",
        23: "B",
        24: "Z",
        25: "-",
        26: ".",
        27: "<mask>",
        28: "<pad>",
    }
    msa_sequence_list = []
    row, col = token_array.shape
    for i in range(row):
        token_list = []
        for k in range(col):
            token = amino_acid_dict[token_array[i, k]]
            token_list.append(token)
        sequence = "".join(token_list)
        msa_sequence_list.append((str(i), sequence))
    return msa_sequence_list


def esm_tokens2seq(token_array):
    # X: rare amino acid or unknown amino acid;  "-": gap;
    amino_acid_dict = {
        4: "L",
        5: "A",
        6: "G",
        7: "V",
        8: "S",
        9: "E",
        10: "R",
        11: "T",
        12: "I",
        13: "D",
        14: "P",
        15: "K",
        16: "Q",
        17: "N",
        18: "F",
        19: "Y",
        20: "M",
        21: "H",
        22: "W",
        23: "C",
        24: "X",
        25: "B",
        26: "U",
        27: "Z",
        28: "O",
        29: ".",
        30: "-",
    }
    msa_sequence_list = []
    row, col = token_array.shape
    for i in range(row):
        token_list = []
        for k in range(col):
            token = amino_acid_dict[token_array[i, k]]
            token_list.append(token)
        sequence = "".join(token_list)
        msa_sequence_list.append((str(i), sequence))
    return msa_sequence_list


def ESM2_embed(seq_tokens, batch_converter_esm2, model_CPU, model_GPU, length_cutoff, gpu):

    batch, L = seq_tokens.shape
    L = L - 2
    with torch.no_grad():
        if L > length_cutoff:
            print("L", L, "length_cutoff", length_cutoff, "CPU")
            results = model_CPU(seq_tokens.to(device="cpu"), repr_layers=range(37), need_head_weights=False, return_contacts=True)
        else:
            print("L", L, "length_cutoff", length_cutoff, "GPU")
            results = model_GPU(seq_tokens.to(device=gpu), repr_layers=range(37), need_head_weights=False, return_contacts=True)

    token_embeds = torch.stack([v for _, v in sorted(results["representations"].items())], dim=2)
    token_embeds = token_embeds[:, 1:-1]  # (batch, L, 37, dim=2560)
    token_embeds = token_embeds.to(device=gpu, dtype=torch.float32)

    ###### attention map and contact map ######
    attentions = results["attentions"]  # (batch, layers=36, heads=40, L+2, L+2)
    attentions = attentions[:, -1, :, 1:-1, 1:-1]  # (batch, 40, L, L)
    contacts = results["contacts"].unsqueeze(1)  # (batch, 1, L, L)
    return (token_embeds, attentions)


def threadTrain(spired_fitess_model, target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, single_index, double_index, single_label, double_label, labelbatch, label_phi_psi, FLAGS, sample_idx, device1_G_E2S):
    # f1d_esm2_3B = (batch, L, 37, embed); f2dbatch = (batch, channel, L, L); labelbatch=(batch,8,L,L)

    L = esm1v_single_logits.shape[-2]
    start_idx = sample_idx % 4
    print("sample_idx", sample_idx)
    coord_idx = list(range(start_idx, L, 4))

    if FLAGS.train_cycle > 1:
        cycle = np.random.randint(low=1, high=FLAGS.train_cycle)
    else:
        cycle = 1

    print("cycle", cycle)
    s0 = time.time()

    if L <= 800:
        print("L", L, "train_length", "SPIRED on GPU")
        single_pred, double_pred, Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, f1d_cycle, f2d_cycle = spired_fitess_model(target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits)
    else:
        # transfer SPIRED to cpu
        print("L", L, "train_length", "SPIRED on CPU")
        target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, spired_fitess_model, single_label, double_label, labelbatch, label_phi_psi = target_tokens.to(device="cpu"), f1d_esm2_3B.to(device="cpu"), f1d_esm2_650M.to(device="cpu"), esm1v_single_logits.to(device="cpu"), esm1v_double_logits.to(device="cpu"), spired_fitess_model.to(device="cpu"), single_label.to(device="cpu"), double_label.to(device="cpu"), labelbatch.to(device="cpu"), label_phi_psi.to(device="cpu")
        single_pred, double_pred, Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, f1d_cycle, f2d_cycle = spired_fitess_model(target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits)

    s1 = time.time()

    # calculate soft spearman loss
    single_label, double_label = single_label.to(single_pred.device), double_label.to(single_pred.device)
    if len(single_index[0]) != 0:
        # calculate single mutation spearman corr
        single_spearman_corr = spearman_corr(single_pred[0][single_index], single_label[0][single_index])
        if len(double_index[0]) != 0:
            print("calculate double mutations spearman corr")
            pred = torch.cat((single_pred[0][single_index], double_pred[0][double_index]), dim=0)
            label = torch.cat((single_label[0][single_index], double_label[0][double_index]), dim=0)
            soft_spearman_loss = Loss.spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), FLAGS.regularization_strength, "kl")
            double_spearman_corr = spearman_corr(double_pred[0][double_index], double_label[0][double_index])
        else:
            print("calculate single mutations spearman corr")
            soft_spearman_loss = Loss.spearman_loss(single_pred[0][single_index].unsqueeze(0), single_label[0][single_index].unsqueeze(0), FLAGS.regularization_strength, "kl")
            double_spearman_corr = torch.tensor(0.0).to(device=single_pred.device)
    else:
        soft_spearman_loss = torch.tensor(0.0).to(device=single_pred.device)
        single_spearman_corr = torch.tensor(0.0).to(device=single_pred.device)
        double_spearman_corr = torch.tensor(0.0).to(device=single_pred.device)

    ## Calculate structure Loss
    labelbatch, label_phi_psi = labelbatch.to(single_pred.device), label_phi_psi.to(single_pred.device)
    Losse2e, lossRD_byResidue = Loss.getLossMultBlock(Predxyz, PredCadistavg, Plddt, labelbatch, coord_idx)
    LossCE = Loss.getLossCE(cb, omega, theta, phi, labelbatch)  # LossDictoptim, cbLossDict, omegaLossDict, thetaLossDict, phiLossDict
    LossCA = Loss.getLossCAL1(ca, labelbatch, cutoff=32)
    loss_phi_psi_1D = Loss.getLoss_phi_psi_1D_L1(phi_psi_1D, label_phi_psi)
    Loss_struct = np.sqrt(L) * (4 * Losse2e["optim"] + 0.01 * LossCE["optim"] + 0.5 * loss_phi_psi_1D)
    print("\n", "Loss_struct_sum", Loss_struct.item(), "soft_spearman_loss", soft_spearman_loss.item(), "LossRD", np.sqrt(L) * 4 * Losse2e["optim"].item(), "LossCE", np.sqrt(L) * 0.01 * LossCE["optim"].item(), "loss_phi_psi_1D", np.sqrt(L) * 0.5 * loss_phi_psi_1D.item(), "\n")

    ## Calculate Fitness Loss
    s2 = time.time()
    t_train1 = s1 - s0
    t_train2 = s2 - s1
    print(
        "threadTrain",
        "train time",
        t_train1,
        "loss time",
        t_train2,
    )

    return (single_pred, double_pred, Predxyz, PredCadistavg, Plddt, phi_psi_1D, soft_spearman_loss, single_spearman_corr, double_spearman_corr, Loss_struct, Losse2e, LossCE, LossCA, loss_phi_psi_1D)


def threadValid(sample, spired_fitess_model, target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, single_index, double_index, single_label, double_label, labelbatch, label_phi_psi, FLAGS, sample_idx, device1_G_E2S):

    L = esm1v_single_logits.shape[-2]

    cycle = FLAGS.valid_cycle
    start_idx = sample_idx % 4

    coord_idx = list(range(start_idx, L, 4))

    s0 = time.time()
    if L < FLAGS.valid_length:
        single_pred, double_pred, Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, f1d_cycle, f2d_cycle = spired_fitess_model(target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits)
    else:
        # transfer data to cpu
        target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, spired_fitess_model, single_label, double_label, labelbatch, label_phi_psi = target_tokens.to(device="cpu"), f1d_esm2_3B.to(device="cpu"), f1d_esm2_650M.to(device="cpu"), esm1v_single_logits.to(device="cpu"), esm1v_double_logits.to(device="cpu"), spired_fitess_model.to(device="cpu"), single_label.to(device="cpu"), double_label.to(device="cpu"), labelbatch.to(device="cpu"), label_phi_psi.to(device="cpu")

        single_pred, double_pred, Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, f1d_cycle, f2d_cycle = spired_fitess_model(target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits)

    s1 = time.time()

    # calculate soft spearman loss
    if len(single_index[0]) != 0:
        single_spearman_corr = spearman_corr(single_pred[0][single_index], single_label[0][single_index])
        if len(double_index[0]) != 0:
            print("calculate double mutations spearman corr")
            pred = torch.cat((single_pred[0][single_index], double_pred[0][double_index]), dim=0)
            label = torch.cat((single_label[0][single_index], double_label[0][double_index]), dim=0)
            soft_spearman_loss = Loss.spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), FLAGS.regularization_strength, "kl")
            double_spearman_corr = spearman_corr(double_pred[0][double_index], double_label[0][double_index])
        else:
            print("calculate single mutations spearman corr")
            soft_spearman_loss = Loss.spearman_loss(single_pred[0][single_index].unsqueeze(0), single_label[0][single_index].unsqueeze(0), FLAGS.regularization_strength, "kl")
            double_spearman_corr = torch.tensor(0.0).to(device=single_pred.device)
    else:
        soft_spearman_loss = torch.tensor(0.0).to(device=single_pred.device)
        single_spearman_corr = torch.tensor(0.0).to(device=single_pred.device)
        double_spearman_corr = torch.tensor(0.0).to(device=single_pred.device)

    ## Calculate Loss
    Losse2e, lossRD_byResidue = Loss.getLossMultBlock(Predxyz, PredCadistavg, Plddt, labelbatch, coord_idx)
    LossCE = Loss.getLossCE(cb, omega, theta, phi, labelbatch)
    LossCA = Loss.getLossCAL1(ca, labelbatch, cutoff=32)
    loss_phi_psi_1D = Loss.getLoss_phi_psi_1D_L1(phi_psi_1D, label_phi_psi)
    # Lossoptim = np.sqrt(L) * (4 * Losse2e['optim'] + 0.1 * LossCE['optim'] + 0.5 * loss_phi_psi_1D)
    Loss_struct = np.sqrt(L) * (4 * Losse2e["optim"] + 0.01 * LossCE["optim"] + 0.5 * loss_phi_psi_1D)

    print("\n", "Loss_struct_sum", Loss_struct.item(), "single_spearman_corr", single_spearman_corr.item(), "LossRD", np.sqrt(L) * 4 * Losse2e["optim"].item(), "LossCE", np.sqrt(L) * 0.01 * LossCE["optim"].item(), "loss_phi_psi_1D", np.sqrt(L) * 0.5 * loss_phi_psi_1D.item(), "\n")

    s2 = time.time()
    t_train1 = s1 - s0
    t_train2 = s2 - s1
    print(
        "threadTrain",
        "train time",
        t_train1,
        "loss time",
        t_train2,
    )

    return (single_pred, double_pred, Predxyz, PredCadistavg, Plddt, phi_psi_1D, soft_spearman_loss, single_spearman_corr, double_spearman_corr, Loss_struct, Losse2e, LossCE, LossCA, loss_phi_psi_1D)


def generate_esm1v_logits(model, batch_converter, alphabet, seq_data):

    with torch.no_grad():

        batch_labels, batch_strs, batch_tokens = batch_converter(seq_data)
        seq = seq_data[0][1]

        token_probs = torch.log_softmax(model(batch_tokens)["logits"], dim=-1)
        logits_33 = token_probs[0, 1:-1, :].detach().cpu().clone()

        # logits 33 dim -> 20 dim
        amino_acid_list = list("ARNDCQEGHILKMFPSTWYV")
        esm1v_amino_acid_dict = {}
        for i in amino_acid_list:
            esm1v_amino_acid_dict[i] = alphabet.get_idx(i)

        logits_20_single = torch.zeros((logits_33.shape[0], 20))
        for wt_pos, wt_amino_acid in enumerate(seq):
            for mut_pos, mut_amino_acid in enumerate(amino_acid_list):
                logits_20_single[wt_pos, mut_pos] = logits_33[wt_pos, esm1v_amino_acid_dict[mut_amino_acid]] - logits_33[wt_pos, esm1v_amino_acid_dict[wt_amino_acid]]

        logits_20_double = (logits_20_single[:, None, :, None] + logits_20_single[None, :, None, :]).reshape(len(seq), len(seq), 20 * 20)
        return (logits_20_single.unsqueeze(0), logits_20_double.unsqueeze(0))


def getDataTest(seq, ESM2_3B, ESM2_650M, ESM1v_1, ESM1v_2, ESM1v_3, ESM1v_4, ESM1v_5, esm1v_batch_converter, esm1v_alphabet, esm2_batch_converter):

    with torch.no_grad():
        _, _, target_tokens = esm2_batch_converter([("", seq)])
        results = ESM2_3B(target_tokens, repr_layers=range(37), need_head_weights=False, return_contacts=False)
        f1d_esm2_3B = torch.stack([v for _, v in sorted(results["representations"].items())], dim=2)
        f1d_esm2_3B = f1d_esm2_3B[:, 1:-1]
        f1d_esm2_3B = f1d_esm2_3B.to(dtype=torch.float32)

        result_esm2_650m = ESM2_650M(target_tokens, repr_layers=[33], return_contacts=False)
        f1d_esm2_650M = result_esm2_650m["representations"][33][0, 1:-1, :].unsqueeze(0)

        esm1v_single_1, esm1v_double_1 = generate_esm1v_logits(ESM1v_1, esm1v_batch_converter, esm1v_alphabet, [("", seq)])
        esm1v_single_2, esm1v_double_2 = generate_esm1v_logits(ESM1v_2, esm1v_batch_converter, esm1v_alphabet, [("", seq)])
        esm1v_single_3, esm1v_double_3 = generate_esm1v_logits(ESM1v_3, esm1v_batch_converter, esm1v_alphabet, [("", seq)])
        esm1v_single_4, esm1v_double_4 = generate_esm1v_logits(ESM1v_4, esm1v_batch_converter, esm1v_alphabet, [("", seq)])
        esm1v_single_5, esm1v_double_5 = generate_esm1v_logits(ESM1v_5, esm1v_batch_converter, esm1v_alphabet, [("", seq)])
        esm1v_single_logits = torch.cat([esm1v_single_1, esm1v_single_2, esm1v_single_3, esm1v_single_4, esm1v_single_5], dim=0).unsqueeze(0)
        esm1v_double_logits = torch.cat([esm1v_double_1, esm1v_double_2, esm1v_double_3, esm1v_double_4, esm1v_double_5], dim=0).unsqueeze(0)

    return f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, target_tokens[:, 1:-1]


def getStabDataTest(seq, ESM2_3B, ESM2_650M, esm2_batch_converter, device="cpu"):

    with torch.no_grad():
        _, _, target_tokens = esm2_batch_converter([("", seq)])
        results = ESM2_3B(target_tokens.to(device), repr_layers=range(37), need_head_weights=False, return_contacts=False)
        f1d_esm2_3B = torch.stack([v for _, v in sorted(results["representations"].items())], dim=2)
        f1d_esm2_3B = f1d_esm2_3B[:, 1:-1]
        f1d_esm2_3B = f1d_esm2_3B.to(dtype=torch.float32)

        result_esm2_650m = ESM2_650M(target_tokens.to(device), repr_layers=[33], return_contacts=False)
        f1d_esm2_650M = result_esm2_650m["representations"][33][0, 1:-1, :].unsqueeze(0)
    return f1d_esm2_3B, f1d_esm2_650M, target_tokens[:, 1:-1]


def threadTest(sample, spired_fitess_model, spired_fitess_model_cpu, target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, FLAGS):

    L = f1d_esm2_3B.shape[2]

    s0 = time.time()

    # transfer data to cpu
    single_pred, double_pred, Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, f1d_cycle, f2d_cycle = spired_fitess_model(target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits)

    s1 = time.time()

    t_train1 = s1 - s0
    print("threadTrain", "infer time", t_train1)

    return (single_pred, double_pred, Predxyz, PredCadistavg, Plddt, phi_psi_1D)


def dxyz_calculate(Predxyz):
    Predxyz_b1 = Predxyz["1st"]
    Predxyz_b2 = Predxyz["2nd"]
    Predxyz_b3 = Predxyz["3rd"]
    Predxyz_b4 = Predxyz["4th"]

    dxyz_b1_0 = Predxyz_b1[0]
    dxyz_b1_0 = torch.sqrt(torch.sum(torch.mean(dxyz_b1_0, dim=0) ** 2, dim=0)).mean()

    dxyz_b1_1 = Predxyz_b1[1] - Predxyz_b1[0]
    dxyz_b1_1 = torch.sqrt(torch.sum(torch.mean(dxyz_b1_1, dim=0) ** 2, dim=0)).mean()

    dxyz_b2_0 = Predxyz_b2[0]
    dxyz_b2_0 = torch.sqrt(torch.sum(torch.mean(dxyz_b2_0, dim=0) ** 2, dim=0)).mean()

    dxyz_b2_1 = Predxyz_b2[1] - Predxyz_b2[0]
    dxyz_b2_1 = torch.sqrt(torch.sum(torch.mean(dxyz_b2_1, dim=0) ** 2, dim=0)).mean()

    dxyz_b3_0 = Predxyz_b3[0]
    dxyz_b3_0 = torch.sqrt(torch.sum(torch.mean(dxyz_b3_0, dim=0) ** 2, dim=0)).mean()

    dxyz_b3_1 = Predxyz_b3[1] - Predxyz_b3[0]
    dxyz_b3_1 = torch.sqrt(torch.sum(torch.mean(dxyz_b3_1, dim=0) ** 2, dim=0)).mean()

    dxyz_b4_0 = Predxyz_b4[0]
    dxyz_b4_0 = torch.sqrt(torch.sum(torch.mean(dxyz_b4_0, dim=0) ** 2, dim=0)).mean()

    dxyz_b4_1 = Predxyz_b4[1] - Predxyz_b4[0]
    dxyz_b4_1 = torch.sqrt(torch.sum(torch.mean(dxyz_b4_1, dim=0) ** 2, dim=0)).mean()

    dxyz_b4_2 = Predxyz_b4[2] - Predxyz_b4[1]
    dxyz_b4_2 = torch.sqrt(torch.sum(torch.mean(dxyz_b4_2, dim=0) ** 2, dim=0)).mean()

    dxyz_b4_3 = Predxyz_b4[3] - Predxyz_b4[2]
    dxyz_b4_3 = torch.sqrt(torch.sum(torch.mean(dxyz_b4_3, dim=0) ** 2, dim=0)).mean()

    dxyz_b4_4 = Predxyz_b4[4] - Predxyz_b4[3]
    dxyz_b4_4 = torch.sqrt(torch.sum(torch.mean(dxyz_b4_4, dim=0) ** 2, dim=0)).mean()

    dxyz_b4_5 = Predxyz_b4[5] - Predxyz_b4[4]
    dxyz_b4_5 = torch.sqrt(torch.sum(torch.mean(dxyz_b4_5, dim=0) ** 2, dim=0)).mean()

    return (dxyz_b1_0, dxyz_b1_1, dxyz_b2_0, dxyz_b2_1, dxyz_b3_0, dxyz_b3_1, dxyz_b4_0, dxyz_b4_1, dxyz_b4_2, dxyz_b4_3, dxyz_b4_4, dxyz_b4_5)


def train_loss_dict(LossDict, Predxyz, Losse2e, LossCE, LossCA, loss_phi_psi_1D):

    LossDict["train/sumloss"].append(Losse2e["optim"].item())
    LossDict["train/cadist"].append(LossCA.item())
    LossDict["train/cbdist"].append(LossCE["cb"].item())
    LossDict["train/omega"].append(LossCE["omega"].item())
    LossDict["train/theta"].append(LossCE["theta"].item())
    LossDict["train/phi"].append(LossCE["phi"].item())
    LossDict["train/phi_psi_1D"].append(loss_phi_psi_1D.item())

    (dxyz_b1_0, dxyz_b1_1, dxyz_b2_0, dxyz_b2_1, dxyz_b3_0, dxyz_b3_1, dxyz_b4_0, dxyz_b4_1, dxyz_b4_2, dxyz_b4_3, dxyz_b4_4, dxyz_b4_5) = dxyz_calculate(Predxyz)
    LossDict["train/dxyz_b1_0"].append(dxyz_b1_0.item())
    LossDict["train/dxyz_b1_1"].append(dxyz_b1_1.item())
    LossDict["train/dxyz_b2_0"].append(dxyz_b2_0.item())
    LossDict["train/dxyz_b2_1"].append(dxyz_b2_1.item())
    LossDict["train/dxyz_b3_0"].append(dxyz_b3_0.item())
    LossDict["train/dxyz_b3_1"].append(dxyz_b3_1.item())
    LossDict["train/dxyz_b4_0"].append(dxyz_b4_0.item())
    LossDict["train/dxyz_b4_1"].append(dxyz_b4_1.item())
    LossDict["train/dxyz_b4_2"].append(dxyz_b4_2.item())
    LossDict["train/dxyz_b4_3"].append(dxyz_b4_3.item())
    LossDict["train/dxyz_b4_4"].append(dxyz_b4_4.item())
    LossDict["train/dxyz_b4_5"].append(dxyz_b4_5.item())

    LossDict["train/1st/RDloss24_1/RDloss24_1"].append(Losse2e["RD"]["1st"]["RDloss24_1"].item())
    LossDict["train/1st/RDloss24_2/RDloss24_2"].append(Losse2e["RD"]["1st"]["RDloss24_2"].item())

    LossDict["train/1st/RDloss24_1/LossCadist"].append(Losse2e["Cadist"]["1st"]["RDloss24_1"].item())
    LossDict["train/1st/RDloss24_2/LossCadist"].append(Losse2e["Cadist"]["1st"]["RDloss24_2"].item())

    LossDict["train/1st/RDloss24_1/realfape"].append(Losse2e["RealFape"]["1st"]["RDloss24_1"].item())
    LossDict["train/1st/RDloss24_2/realfape"].append(Losse2e["RealFape"]["1st"]["RDloss24_2"].item())

    LossDict["train/1st/RDloss24_1/realCadistLoss"].append(Losse2e["RealCadist"]["1st"]["RDloss24_1"].item())
    LossDict["train/1st/RDloss24_2/realCadistLoss"].append(Losse2e["RealCadist"]["1st"]["RDloss24_2"].item())

    LossDict["train/1st/RDloss24_2/plddt_loss"].append(Losse2e["plddt_loss"]["1st"]["RDloss24_2"].item())

    truelddt = Losse2e["truelddt"]["1st"]["RDloss24_2"]
    truelddt = torch.mean(torch.mean(truelddt, dim=0), dim=1)
    LossDict["train/1st/RDloss24_2/truelddtmax"].append(torch.max(truelddt).item())
    LossDict["train/1st/RDloss24_2/truelddtmean"].append(torch.mean(truelddt).item())
    LossDict["train/1st/RDloss24_2/truelddtmedian"].append(torch.median(truelddt).item())

    plddt = Losse2e["plddt"]["1st"]["RDloss24_2"]
    plddt = torch.mean(torch.mean(plddt, dim=0), dim=1)
    LossDict["train/1st/RDloss24_2/plddtmax"].append(torch.max(plddt).item())
    LossDict["train/1st/RDloss24_2/plddtmean"].append(torch.mean(plddt).item())
    LossDict["train/1st/RDloss24_2/plddtmedian"].append(torch.median(plddt).item())

    LossDict["train/2nd/RDloss24_1/RDloss24_1"].append(Losse2e["RD"]["2nd"]["RDloss24_1"].item())
    LossDict["train/2nd/RDloss24_2/RDloss24_2"].append(Losse2e["RD"]["2nd"]["RDloss24_2"].item())

    LossDict["train/2nd/RDloss24_1/LossCadist"].append(Losse2e["Cadist"]["2nd"]["RDloss24_1"].item())
    LossDict["train/2nd/RDloss24_2/LossCadist"].append(Losse2e["Cadist"]["2nd"]["RDloss24_2"].item())

    LossDict["train/2nd/RDloss24_1/realfape"].append(Losse2e["RealFape"]["2nd"]["RDloss24_1"].item())
    LossDict["train/2nd/RDloss24_2/realfape"].append(Losse2e["RealFape"]["2nd"]["RDloss24_2"].item())

    LossDict["train/2nd/RDloss24_1/realCadistLoss"].append(Losse2e["RealCadist"]["2nd"]["RDloss24_1"].item())
    LossDict["train/2nd/RDloss24_2/realCadistLoss"].append(Losse2e["RealCadist"]["2nd"]["RDloss24_2"].item())

    LossDict["train/2nd/RDloss24_2/plddt_loss"].append(Losse2e["plddt_loss"]["2nd"]["RDloss24_2"].item())

    truelddt = Losse2e["truelddt"]["2nd"]["RDloss24_2"]
    truelddt = torch.mean(torch.mean(truelddt, dim=0), dim=1)
    LossDict["train/2nd/RDloss24_2/truelddtmax"].append(torch.max(truelddt).item())
    LossDict["train/2nd/RDloss24_2/truelddtmean"].append(torch.mean(truelddt).item())
    LossDict["train/2nd/RDloss24_2/truelddtmedian"].append(torch.median(truelddt).item())

    plddt = Losse2e["plddt"]["2nd"]["RDloss24_2"]
    plddt = torch.mean(torch.mean(plddt, dim=0), dim=1)
    LossDict["train/2nd/RDloss24_2/plddtmax"].append(torch.max(plddt).item())
    LossDict["train/2nd/RDloss24_2/plddtmean"].append(torch.mean(plddt).item())
    LossDict["train/2nd/RDloss24_2/plddtmedian"].append(torch.median(plddt).item())

    LossDict["train/3rd/RDloss24_1/RDloss24_1"].append(Losse2e["RD"]["3rd"]["RDloss24_1"].item())
    LossDict["train/3rd/RDloss24_2/RDloss24_2"].append(Losse2e["RD"]["3rd"]["RDloss24_2"].item())

    LossDict["train/3rd/RDloss24_1/LossCadist"].append(Losse2e["Cadist"]["3rd"]["RDloss24_1"].item())
    LossDict["train/3rd/RDloss24_2/LossCadist"].append(Losse2e["Cadist"]["3rd"]["RDloss24_2"].item())

    LossDict["train/3rd/RDloss24_1/realfape"].append(Losse2e["RealFape"]["3rd"]["RDloss24_1"].item())
    LossDict["train/3rd/RDloss24_2/realfape"].append(Losse2e["RealFape"]["3rd"]["RDloss24_2"].item())

    LossDict["train/3rd/RDloss24_1/realCadistLoss"].append(Losse2e["RealCadist"]["3rd"]["RDloss24_1"].item())
    LossDict["train/3rd/RDloss24_2/realCadistLoss"].append(Losse2e["RealCadist"]["3rd"]["RDloss24_2"].item())

    LossDict["train/3rd/RDloss24_2/plddt_loss"].append(Losse2e["plddt_loss"]["3rd"]["RDloss24_2"].item())

    truelddt = Losse2e["truelddt"]["3rd"]["RDloss24_2"]
    truelddt = torch.mean(torch.mean(truelddt, dim=0), dim=1)
    LossDict["train/3rd/RDloss24_2/truelddtmax"].append(torch.max(truelddt).item())
    LossDict["train/3rd/RDloss24_2/truelddtmean"].append(torch.mean(truelddt).item())
    LossDict["train/3rd/RDloss24_2/truelddtmedian"].append(torch.median(truelddt).item())

    plddt = Losse2e["plddt"]["3rd"]["RDloss24_2"]
    plddt = torch.mean(torch.mean(plddt, dim=0), dim=1)
    LossDict["train/3rd/RDloss24_2/plddtmax"].append(torch.max(plddt).item())
    LossDict["train/3rd/RDloss24_2/plddtmean"].append(torch.mean(plddt).item())
    LossDict["train/3rd/RDloss24_2/plddtmedian"].append(torch.median(plddt).item())

    LossDict["train/4th/RDloss32_1/RDloss32_1"].append(Losse2e["RD"]["4th"]["RDloss32_1"].item())
    LossDict["train/4th/RDloss32_2/RDloss32_2"].append(Losse2e["RD"]["4th"]["RDloss32_2"].item())
    LossDict["train/4th/RDloss32_3/RDloss32_3"].append(Losse2e["RD"]["4th"]["RDloss32_3"].item())
    LossDict["train/4th/RDloss32_4/RDloss32_4"].append(Losse2e["RD"]["4th"]["RDloss32_4"].item())
    LossDict["train/4th/fape40_1/fapeloss"].append(Losse2e["Fape"]["4th"]["fape40_1"].item())
    LossDict["train/4th/fape40_2/fapeloss"].append(Losse2e["Fape"]["4th"]["fape40_2"].item())

    LossDict["train/4th/RDloss32_1/LossCadist"].append(Losse2e["Cadist"]["4th"]["RDloss32_1"].item())
    LossDict["train/4th/RDloss32_2/LossCadist"].append(Losse2e["Cadist"]["4th"]["RDloss32_2"].item())
    LossDict["train/4th/RDloss32_3/LossCadist"].append(Losse2e["Cadist"]["4th"]["RDloss32_3"].item())
    LossDict["train/4th/RDloss32_4/LossCadist"].append(Losse2e["Cadist"]["4th"]["RDloss32_4"].item())
    LossDict["train/4th/fape40_1/LossCadist"].append(Losse2e["Cadist"]["4th"]["fape40_1"].item())
    LossDict["train/4th/fape40_2/LossCadist"].append(Losse2e["Cadist"]["4th"]["fape40_2"].item())

    LossDict["train/4th/RDloss32_1/realfape"].append(Losse2e["RealFape"]["4th"]["RDloss32_1"].item())
    LossDict["train/4th/RDloss32_2/realfape"].append(Losse2e["RealFape"]["4th"]["RDloss32_2"].item())
    LossDict["train/4th/RDloss32_3/realfape"].append(Losse2e["RealFape"]["4th"]["RDloss32_3"].item())
    LossDict["train/4th/RDloss32_4/realfape"].append(Losse2e["RealFape"]["4th"]["RDloss32_4"].item())
    LossDict["train/4th/fape40_1/realfape"].append(Losse2e["RealFape"]["4th"]["fape40_1"].item())
    LossDict["train/4th/fape40_2/realfape"].append(Losse2e["RealFape"]["4th"]["fape40_2"].item())

    LossDict["train/4th/RDloss32_1/realCadistLoss"].append(Losse2e["RealCadist"]["4th"]["RDloss32_1"].item())
    LossDict["train/4th/RDloss32_2/realCadistLoss"].append(Losse2e["RealCadist"]["4th"]["RDloss32_2"].item())
    LossDict["train/4th/RDloss32_3/realCadistLoss"].append(Losse2e["RealCadist"]["4th"]["RDloss32_3"].item())
    LossDict["train/4th/RDloss32_4/realCadistLoss"].append(Losse2e["RealCadist"]["4th"]["RDloss32_4"].item())
    LossDict["train/4th/fape40_1/realCadistLoss"].append(Losse2e["RealCadist"]["4th"]["fape40_1"].item())
    LossDict["train/4th/fape40_2/realCadistLoss"].append(Losse2e["RealCadist"]["4th"]["fape40_2"].item())

    LossDict["train/4th/fape40_2/plddt_loss"].append(Losse2e["plddt_loss"]["4th"]["fape40_2"].item())

    truelddt = Losse2e["truelddt"]["4th"]["fape40_2"]
    truelddt = torch.mean(torch.mean(truelddt, dim=0), dim=1)
    LossDict["train/4th/fape40_2/truelddtmax"].append(torch.max(truelddt).item())
    LossDict["train/4th/fape40_2/truelddtmean"].append(torch.mean(truelddt).item())
    LossDict["train/4th/fape40_2/truelddtmedian"].append(torch.median(truelddt).item())

    plddt = Losse2e["plddt"]["4th"]["fape40_2"]
    plddt = torch.mean(torch.mean(plddt, dim=0), dim=1)
    LossDict["train/4th/fape40_2/plddtmax"].append(torch.max(plddt).item())
    LossDict["train/4th/fape40_2/plddtmean"].append(torch.mean(plddt).item())
    LossDict["train/4th/fape40_2/plddtmedian"].append(torch.median(plddt).item())

    return LossDict


def valid_loss_dict(LossDict, Predxyz, Losse2e, LossCE, LossCA, loss_phi_psi_1D):

    LossDict["valid/sumloss"].append(Losse2e["optim"].item())
    LossDict["valid/cadist"].append(LossCA.item())
    LossDict["valid/cbdist"].append(LossCE["cb"].item())
    LossDict["valid/omega"].append(LossCE["omega"].item())
    LossDict["valid/theta"].append(LossCE["theta"].item())
    LossDict["valid/phi"].append(LossCE["phi"].item())
    LossDict["valid/phi_psi_1D"].append(loss_phi_psi_1D.item())

    (dxyz_b1_0, dxyz_b1_1, dxyz_b2_0, dxyz_b2_1, dxyz_b3_0, dxyz_b3_1, dxyz_b4_0, dxyz_b4_1, dxyz_b4_2, dxyz_b4_3, dxyz_b4_4, dxyz_b4_5) = dxyz_calculate(Predxyz)
    LossDict["valid/dxyz_b1_0"].append(dxyz_b1_0.item())
    LossDict["valid/dxyz_b1_1"].append(dxyz_b1_1.item())
    LossDict["valid/dxyz_b2_0"].append(dxyz_b2_0.item())
    LossDict["valid/dxyz_b2_1"].append(dxyz_b2_1.item())
    LossDict["valid/dxyz_b3_0"].append(dxyz_b3_0.item())
    LossDict["valid/dxyz_b3_1"].append(dxyz_b3_1.item())
    LossDict["valid/dxyz_b4_0"].append(dxyz_b4_0.item())
    LossDict["valid/dxyz_b4_1"].append(dxyz_b4_1.item())
    LossDict["valid/dxyz_b4_2"].append(dxyz_b4_2.item())
    LossDict["valid/dxyz_b4_3"].append(dxyz_b4_3.item())
    LossDict["valid/dxyz_b4_4"].append(dxyz_b4_4.item())
    LossDict["valid/dxyz_b4_5"].append(dxyz_b4_5.item())

    LossDict["valid/1st/RDloss24_1/RDloss24_1"].append(Losse2e["RD"]["1st"]["RDloss24_1"].item())
    LossDict["valid/1st/RDloss24_2/RDloss24_2"].append(Losse2e["RD"]["1st"]["RDloss24_2"].item())

    LossDict["valid/1st/RDloss24_1/LossCadist"].append(Losse2e["Cadist"]["1st"]["RDloss24_1"].item())
    LossDict["valid/1st/RDloss24_2/LossCadist"].append(Losse2e["Cadist"]["1st"]["RDloss24_2"].item())

    LossDict["valid/1st/RDloss24_1/realfape"].append(Losse2e["RealFape"]["1st"]["RDloss24_1"].item())
    LossDict["valid/1st/RDloss24_2/realfape"].append(Losse2e["RealFape"]["1st"]["RDloss24_2"].item())

    LossDict["valid/1st/RDloss24_1/realCadistLoss"].append(Losse2e["RealCadist"]["1st"]["RDloss24_1"].item())
    LossDict["valid/1st/RDloss24_2/realCadistLoss"].append(Losse2e["RealCadist"]["1st"]["RDloss24_2"].item())

    LossDict["valid/1st/RDloss24_2/plddt_loss"].append(Losse2e["plddt_loss"]["1st"]["RDloss24_2"].item())

    truelddt = Losse2e["truelddt"]["1st"]["RDloss24_2"]
    truelddt = torch.mean(torch.mean(truelddt, dim=0), dim=1)
    LossDict["valid/1st/RDloss24_2/truelddtmax"].append(torch.max(truelddt).item())
    LossDict["valid/1st/RDloss24_2/truelddtmean"].append(torch.mean(truelddt).item())
    LossDict["valid/1st/RDloss24_2/truelddtmedian"].append(torch.median(truelddt).item())

    plddt = Losse2e["plddt"]["1st"]["RDloss24_2"]
    plddt = torch.mean(torch.mean(plddt, dim=0), dim=1)
    LossDict["valid/1st/RDloss24_2/plddtmax"].append(torch.max(plddt).item())
    LossDict["valid/1st/RDloss24_2/plddtmean"].append(torch.mean(plddt).item())
    LossDict["valid/1st/RDloss24_2/plddtmedian"].append(torch.median(plddt).item())

    LossDict["valid/2nd/RDloss24_1/RDloss24_1"].append(Losse2e["RD"]["2nd"]["RDloss24_1"].item())
    LossDict["valid/2nd/RDloss24_2/RDloss24_2"].append(Losse2e["RD"]["2nd"]["RDloss24_2"].item())

    LossDict["valid/2nd/RDloss24_1/LossCadist"].append(Losse2e["Cadist"]["2nd"]["RDloss24_1"].item())
    LossDict["valid/2nd/RDloss24_2/LossCadist"].append(Losse2e["Cadist"]["2nd"]["RDloss24_2"].item())

    LossDict["valid/2nd/RDloss24_1/realfape"].append(Losse2e["RealFape"]["2nd"]["RDloss24_1"].item())
    LossDict["valid/2nd/RDloss24_2/realfape"].append(Losse2e["RealFape"]["2nd"]["RDloss24_2"].item())

    LossDict["valid/2nd/RDloss24_1/realCadistLoss"].append(Losse2e["RealCadist"]["2nd"]["RDloss24_1"].item())
    LossDict["valid/2nd/RDloss24_2/realCadistLoss"].append(Losse2e["RealCadist"]["2nd"]["RDloss24_2"].item())

    LossDict["valid/2nd/RDloss24_2/plddt_loss"].append(Losse2e["plddt_loss"]["2nd"]["RDloss24_2"].item())

    truelddt = Losse2e["truelddt"]["2nd"]["RDloss24_2"]
    truelddt = torch.mean(torch.mean(truelddt, dim=0), dim=1)
    LossDict["valid/2nd/RDloss24_2/truelddtmax"].append(torch.max(truelddt).item())
    LossDict["valid/2nd/RDloss24_2/truelddtmean"].append(torch.mean(truelddt).item())
    LossDict["valid/2nd/RDloss24_2/truelddtmedian"].append(torch.median(truelddt).item())

    plddt = Losse2e["plddt"]["2nd"]["RDloss24_2"]
    plddt = torch.mean(torch.mean(plddt, dim=0), dim=1)
    LossDict["valid/2nd/RDloss24_2/plddtmax"].append(torch.max(plddt).item())
    LossDict["valid/2nd/RDloss24_2/plddtmean"].append(torch.mean(plddt).item())
    LossDict["valid/2nd/RDloss24_2/plddtmedian"].append(torch.median(plddt).item())

    LossDict["valid/3rd/RDloss24_1/RDloss24_1"].append(Losse2e["RD"]["3rd"]["RDloss24_1"].item())
    LossDict["valid/3rd/RDloss24_2/RDloss24_2"].append(Losse2e["RD"]["3rd"]["RDloss24_2"].item())

    LossDict["valid/3rd/RDloss24_1/LossCadist"].append(Losse2e["Cadist"]["3rd"]["RDloss24_1"].item())
    LossDict["valid/3rd/RDloss24_2/LossCadist"].append(Losse2e["Cadist"]["3rd"]["RDloss24_2"].item())

    LossDict["valid/3rd/RDloss24_1/realfape"].append(Losse2e["RealFape"]["3rd"]["RDloss24_1"].item())
    LossDict["valid/3rd/RDloss24_2/realfape"].append(Losse2e["RealFape"]["3rd"]["RDloss24_2"].item())

    LossDict["valid/3rd/RDloss24_1/realCadistLoss"].append(Losse2e["RealCadist"]["3rd"]["RDloss24_1"].item())
    LossDict["valid/3rd/RDloss24_2/realCadistLoss"].append(Losse2e["RealCadist"]["3rd"]["RDloss24_2"].item())

    LossDict["valid/3rd/RDloss24_2/plddt_loss"].append(Losse2e["plddt_loss"]["3rd"]["RDloss24_2"].item())

    truelddt = Losse2e["truelddt"]["3rd"]["RDloss24_2"]
    truelddt = torch.mean(torch.mean(truelddt, dim=0), dim=1)
    LossDict["valid/3rd/RDloss24_2/truelddtmax"].append(torch.max(truelddt).item())
    LossDict["valid/3rd/RDloss24_2/truelddtmean"].append(torch.mean(truelddt).item())
    LossDict["valid/3rd/RDloss24_2/truelddtmedian"].append(torch.median(truelddt).item())

    plddt = Losse2e["plddt"]["3rd"]["RDloss24_2"]
    plddt = torch.mean(torch.mean(plddt, dim=0), dim=1)
    LossDict["valid/3rd/RDloss24_2/plddtmax"].append(torch.max(plddt).item())
    LossDict["valid/3rd/RDloss24_2/plddtmean"].append(torch.mean(plddt).item())
    LossDict["valid/3rd/RDloss24_2/plddtmedian"].append(torch.median(plddt).item())

    LossDict["valid/4th/RDloss32_1/RDloss32_1"].append(Losse2e["RD"]["4th"]["RDloss32_1"].item())
    LossDict["valid/4th/RDloss32_2/RDloss32_2"].append(Losse2e["RD"]["4th"]["RDloss32_2"].item())
    LossDict["valid/4th/RDloss32_3/RDloss32_3"].append(Losse2e["RD"]["4th"]["RDloss32_3"].item())
    LossDict["valid/4th/RDloss32_4/RDloss32_4"].append(Losse2e["RD"]["4th"]["RDloss32_4"].item())
    LossDict["valid/4th/fape40_1/fapeloss"].append(Losse2e["Fape"]["4th"]["fape40_1"].item())
    LossDict["valid/4th/fape40_2/fapeloss"].append(Losse2e["Fape"]["4th"]["fape40_2"].item())

    LossDict["valid/4th/RDloss32_1/LossCadist"].append(Losse2e["Cadist"]["4th"]["RDloss32_1"].item())
    LossDict["valid/4th/RDloss32_2/LossCadist"].append(Losse2e["Cadist"]["4th"]["RDloss32_2"].item())
    LossDict["valid/4th/RDloss32_3/LossCadist"].append(Losse2e["Cadist"]["4th"]["RDloss32_3"].item())
    LossDict["valid/4th/RDloss32_4/LossCadist"].append(Losse2e["Cadist"]["4th"]["RDloss32_4"].item())
    LossDict["valid/4th/fape40_1/LossCadist"].append(Losse2e["Cadist"]["4th"]["fape40_1"].item())
    LossDict["valid/4th/fape40_2/LossCadist"].append(Losse2e["Cadist"]["4th"]["fape40_2"].item())

    LossDict["valid/4th/RDloss32_1/realfape"].append(Losse2e["RealFape"]["4th"]["RDloss32_1"].item())
    LossDict["valid/4th/RDloss32_2/realfape"].append(Losse2e["RealFape"]["4th"]["RDloss32_2"].item())
    LossDict["valid/4th/RDloss32_3/realfape"].append(Losse2e["RealFape"]["4th"]["RDloss32_3"].item())
    LossDict["valid/4th/RDloss32_4/realfape"].append(Losse2e["RealFape"]["4th"]["RDloss32_4"].item())
    LossDict["valid/4th/fape40_1/realfape"].append(Losse2e["RealFape"]["4th"]["fape40_1"].item())
    LossDict["valid/4th/fape40_2/realfape"].append(Losse2e["RealFape"]["4th"]["fape40_2"].item())

    LossDict["valid/4th/RDloss32_1/realCadistLoss"].append(Losse2e["RealCadist"]["4th"]["RDloss32_1"].item())
    LossDict["valid/4th/RDloss32_2/realCadistLoss"].append(Losse2e["RealCadist"]["4th"]["RDloss32_2"].item())
    LossDict["valid/4th/RDloss32_3/realCadistLoss"].append(Losse2e["RealCadist"]["4th"]["RDloss32_3"].item())
    LossDict["valid/4th/RDloss32_4/realCadistLoss"].append(Losse2e["RealCadist"]["4th"]["RDloss32_4"].item())
    LossDict["valid/4th/fape40_1/realCadistLoss"].append(Losse2e["RealCadist"]["4th"]["fape40_1"].item())
    LossDict["valid/4th/fape40_2/realCadistLoss"].append(Losse2e["RealCadist"]["4th"]["fape40_2"].item())

    LossDict["valid/4th/fape40_2/plddt_loss"].append(Losse2e["plddt_loss"]["4th"]["fape40_2"].item())

    truelddt = Losse2e["truelddt"]["4th"]["fape40_2"]
    truelddt = torch.mean(torch.mean(truelddt, dim=0), dim=1)
    LossDict["valid/4th/fape40_2/truelddtmax"].append(torch.max(truelddt).item())
    LossDict["valid/4th/fape40_2/truelddtmean"].append(torch.mean(truelddt).item())
    LossDict["valid/4th/fape40_2/truelddtmedian"].append(torch.median(truelddt).item())

    plddt = Losse2e["plddt"]["4th"]["fape40_2"]
    plddt = torch.mean(torch.mean(plddt, dim=0), dim=1)
    LossDict["valid/4th/fape40_2/plddtmax"].append(torch.max(plddt).item())
    LossDict["valid/4th/fape40_2/plddtmean"].append(torch.mean(plddt).item())
    LossDict["valid/4th/fape40_2/plddtmedian"].append(torch.median(plddt).item())

    return LossDict


def make_loss_dict():

    LossDict = {}
    LossDict["train/sumloss"] = []
    LossDict["train/cadist"] = []
    LossDict["train/cbdist"] = []
    LossDict["train/omega"] = []
    LossDict["train/theta"] = []
    LossDict["train/phi"] = []
    LossDict["train/phi_psi_1D"] = []

    LossDict["train/dxyz_b1_0"] = []
    LossDict["train/dxyz_b1_1"] = []
    LossDict["train/dxyz_b2_0"] = []
    LossDict["train/dxyz_b2_1"] = []
    LossDict["train/dxyz_b3_0"] = []
    LossDict["train/dxyz_b3_1"] = []
    LossDict["train/dxyz_b4_0"] = []
    LossDict["train/dxyz_b4_1"] = []
    LossDict["train/dxyz_b4_2"] = []
    LossDict["train/dxyz_b4_3"] = []
    LossDict["train/dxyz_b4_4"] = []
    LossDict["train/dxyz_b4_5"] = []

    LossDict["train/1st/RDloss24_1/RDloss24_1"] = []
    LossDict["train/1st/RDloss24_2/RDloss24_2"] = []

    LossDict["train/1st/RDloss24_1/LossCadist"] = []
    LossDict["train/1st/RDloss24_2/LossCadist"] = []

    LossDict["train/1st/RDloss24_1/realfape"] = []
    LossDict["train/1st/RDloss24_2/realfape"] = []

    LossDict["train/1st/RDloss24_1/realCadistLoss"] = []
    LossDict["train/1st/RDloss24_2/realCadistLoss"] = []

    LossDict["train/1st/RDloss24_2/plddt_loss"] = []

    LossDict["train/1st/RDloss24_2/truelddtmax"] = []
    LossDict["train/1st/RDloss24_2/truelddtmean"] = []
    LossDict["train/1st/RDloss24_2/truelddtmedian"] = []

    LossDict["train/1st/RDloss24_2/plddtmax"] = []
    LossDict["train/1st/RDloss24_2/plddtmean"] = []
    LossDict["train/1st/RDloss24_2/plddtmedian"] = []

    LossDict["train/2nd/RDloss24_1/RDloss24_1"] = []
    LossDict["train/2nd/RDloss24_2/RDloss24_2"] = []

    LossDict["train/2nd/RDloss24_1/LossCadist"] = []
    LossDict["train/2nd/RDloss24_2/LossCadist"] = []

    LossDict["train/2nd/RDloss24_1/realfape"] = []
    LossDict["train/2nd/RDloss24_2/realfape"] = []

    LossDict["train/2nd/RDloss24_1/realCadistLoss"] = []
    LossDict["train/2nd/RDloss24_2/realCadistLoss"] = []

    LossDict["train/2nd/RDloss24_2/plddt_loss"] = []

    LossDict["train/2nd/RDloss24_2/truelddtmax"] = []
    LossDict["train/2nd/RDloss24_2/truelddtmean"] = []
    LossDict["train/2nd/RDloss24_2/truelddtmedian"] = []

    LossDict["train/2nd/RDloss24_2/plddtmax"] = []
    LossDict["train/2nd/RDloss24_2/plddtmean"] = []
    LossDict["train/2nd/RDloss24_2/plddtmedian"] = []

    LossDict["train/3rd/RDloss24_1/RDloss24_1"] = []
    LossDict["train/3rd/RDloss24_2/RDloss24_2"] = []

    LossDict["train/3rd/RDloss24_1/LossCadist"] = []
    LossDict["train/3rd/RDloss24_2/LossCadist"] = []

    LossDict["train/3rd/RDloss24_1/realfape"] = []
    LossDict["train/3rd/RDloss24_2/realfape"] = []

    LossDict["train/3rd/RDloss24_1/realCadistLoss"] = []
    LossDict["train/3rd/RDloss24_2/realCadistLoss"] = []

    LossDict["train/3rd/RDloss24_2/plddt_loss"] = []

    LossDict["train/3rd/RDloss24_2/truelddtmax"] = []
    LossDict["train/3rd/RDloss24_2/truelddtmean"] = []
    LossDict["train/3rd/RDloss24_2/truelddtmedian"] = []

    LossDict["train/3rd/RDloss24_2/plddtmax"] = []
    LossDict["train/3rd/RDloss24_2/plddtmean"] = []
    LossDict["train/3rd/RDloss24_2/plddtmedian"] = []

    LossDict["train/4th/RDloss32_1/RDloss32_1"] = []
    LossDict["train/4th/RDloss32_2/RDloss32_2"] = []
    LossDict["train/4th/RDloss32_3/RDloss32_3"] = []
    LossDict["train/4th/RDloss32_4/RDloss32_4"] = []
    LossDict["train/4th/fape40_1/fapeloss"] = []
    LossDict["train/4th/fape40_2/fapeloss"] = []

    LossDict["train/4th/RDloss32_1/LossCadist"] = []
    LossDict["train/4th/RDloss32_2/LossCadist"] = []
    LossDict["train/4th/RDloss32_3/LossCadist"] = []
    LossDict["train/4th/RDloss32_4/LossCadist"] = []
    LossDict["train/4th/fape40_1/LossCadist"] = []
    LossDict["train/4th/fape40_2/LossCadist"] = []

    LossDict["train/4th/RDloss32_1/realfape"] = []
    LossDict["train/4th/RDloss32_2/realfape"] = []
    LossDict["train/4th/RDloss32_3/realfape"] = []
    LossDict["train/4th/RDloss32_4/realfape"] = []
    LossDict["train/4th/fape40_1/realfape"] = []
    LossDict["train/4th/fape40_2/realfape"] = []

    LossDict["train/4th/RDloss32_1/realCadistLoss"] = []
    LossDict["train/4th/RDloss32_2/realCadistLoss"] = []
    LossDict["train/4th/RDloss32_3/realCadistLoss"] = []
    LossDict["train/4th/RDloss32_4/realCadistLoss"] = []
    LossDict["train/4th/fape40_1/realCadistLoss"] = []
    LossDict["train/4th/fape40_2/realCadistLoss"] = []

    LossDict["train/4th/fape40_2/plddt_loss"] = []

    LossDict["train/4th/fape40_2/truelddtmax"] = []
    LossDict["train/4th/fape40_2/truelddtmean"] = []
    LossDict["train/4th/fape40_2/truelddtmedian"] = []

    LossDict["train/4th/fape40_2/plddtmax"] = []
    LossDict["train/4th/fape40_2/plddtmean"] = []
    LossDict["train/4th/fape40_2/plddtmedian"] = []

    ## validation

    LossDict["valid/sumloss"] = []
    LossDict["valid/cadist"] = []
    LossDict["valid/cbdist"] = []
    LossDict["valid/omega"] = []
    LossDict["valid/theta"] = []
    LossDict["valid/phi"] = []
    LossDict["valid/phi_psi_1D"] = []

    LossDict["valid/dxyz_b1_0"] = []
    LossDict["valid/dxyz_b1_1"] = []
    LossDict["valid/dxyz_b2_0"] = []
    LossDict["valid/dxyz_b2_1"] = []
    LossDict["valid/dxyz_b3_0"] = []
    LossDict["valid/dxyz_b3_1"] = []
    LossDict["valid/dxyz_b4_0"] = []
    LossDict["valid/dxyz_b4_1"] = []
    LossDict["valid/dxyz_b4_2"] = []
    LossDict["valid/dxyz_b4_3"] = []
    LossDict["valid/dxyz_b4_4"] = []
    LossDict["valid/dxyz_b4_5"] = []

    LossDict["valid/1st/RDloss24_1/RDloss24_1"] = []
    LossDict["valid/1st/RDloss24_2/RDloss24_2"] = []

    LossDict["valid/1st/RDloss24_1/LossCadist"] = []
    LossDict["valid/1st/RDloss24_2/LossCadist"] = []

    LossDict["valid/1st/RDloss24_1/realfape"] = []
    LossDict["valid/1st/RDloss24_2/realfape"] = []

    LossDict["valid/1st/RDloss24_1/realCadistLoss"] = []
    LossDict["valid/1st/RDloss24_2/realCadistLoss"] = []

    LossDict["valid/1st/RDloss24_2/plddt_loss"] = []

    LossDict["valid/1st/RDloss24_2/truelddtmax"] = []
    LossDict["valid/1st/RDloss24_2/truelddtmean"] = []
    LossDict["valid/1st/RDloss24_2/truelddtmedian"] = []

    LossDict["valid/1st/RDloss24_2/plddtmax"] = []
    LossDict["valid/1st/RDloss24_2/plddtmean"] = []
    LossDict["valid/1st/RDloss24_2/plddtmedian"] = []

    LossDict["valid/2nd/RDloss24_1/RDloss24_1"] = []
    LossDict["valid/2nd/RDloss24_2/RDloss24_2"] = []

    LossDict["valid/2nd/RDloss24_1/LossCadist"] = []
    LossDict["valid/2nd/RDloss24_2/LossCadist"] = []

    LossDict["valid/2nd/RDloss24_1/realfape"] = []
    LossDict["valid/2nd/RDloss24_2/realfape"] = []

    LossDict["valid/2nd/RDloss24_1/realCadistLoss"] = []
    LossDict["valid/2nd/RDloss24_2/realCadistLoss"] = []

    LossDict["valid/2nd/RDloss24_2/plddt_loss"] = []

    LossDict["valid/2nd/RDloss24_2/truelddtmax"] = []
    LossDict["valid/2nd/RDloss24_2/truelddtmean"] = []
    LossDict["valid/2nd/RDloss24_2/truelddtmedian"] = []

    LossDict["valid/2nd/RDloss24_2/plddtmax"] = []
    LossDict["valid/2nd/RDloss24_2/plddtmean"] = []
    LossDict["valid/2nd/RDloss24_2/plddtmedian"] = []

    LossDict["valid/3rd/RDloss24_1/RDloss24_1"] = []
    LossDict["valid/3rd/RDloss24_2/RDloss24_2"] = []

    LossDict["valid/3rd/RDloss24_1/LossCadist"] = []
    LossDict["valid/3rd/RDloss24_2/LossCadist"] = []

    LossDict["valid/3rd/RDloss24_1/realfape"] = []
    LossDict["valid/3rd/RDloss24_2/realfape"] = []

    LossDict["valid/3rd/RDloss24_1/realCadistLoss"] = []
    LossDict["valid/3rd/RDloss24_2/realCadistLoss"] = []

    LossDict["valid/3rd/RDloss24_2/plddt_loss"] = []

    LossDict["valid/3rd/RDloss24_2/truelddtmax"] = []
    LossDict["valid/3rd/RDloss24_2/truelddtmean"] = []
    LossDict["valid/3rd/RDloss24_2/truelddtmedian"] = []

    LossDict["valid/3rd/RDloss24_2/plddtmax"] = []
    LossDict["valid/3rd/RDloss24_2/plddtmean"] = []
    LossDict["valid/3rd/RDloss24_2/plddtmedian"] = []

    LossDict["valid/4th/RDloss32_1/RDloss32_1"] = []
    LossDict["valid/4th/RDloss32_2/RDloss32_2"] = []
    LossDict["valid/4th/RDloss32_3/RDloss32_3"] = []
    LossDict["valid/4th/RDloss32_4/RDloss32_4"] = []
    LossDict["valid/4th/fape40_1/fapeloss"] = []
    LossDict["valid/4th/fape40_2/fapeloss"] = []

    LossDict["valid/4th/RDloss32_1/LossCadist"] = []
    LossDict["valid/4th/RDloss32_2/LossCadist"] = []
    LossDict["valid/4th/RDloss32_3/LossCadist"] = []
    LossDict["valid/4th/RDloss32_4/LossCadist"] = []
    LossDict["valid/4th/fape40_1/LossCadist"] = []
    LossDict["valid/4th/fape40_2/LossCadist"] = []

    LossDict["valid/4th/RDloss32_1/realfape"] = []
    LossDict["valid/4th/RDloss32_2/realfape"] = []
    LossDict["valid/4th/RDloss32_3/realfape"] = []
    LossDict["valid/4th/RDloss32_4/realfape"] = []
    LossDict["valid/4th/fape40_1/realfape"] = []
    LossDict["valid/4th/fape40_2/realfape"] = []

    LossDict["valid/4th/RDloss32_1/realCadistLoss"] = []
    LossDict["valid/4th/RDloss32_2/realCadistLoss"] = []
    LossDict["valid/4th/RDloss32_3/realCadistLoss"] = []
    LossDict["valid/4th/RDloss32_4/realCadistLoss"] = []
    LossDict["valid/4th/fape40_1/realCadistLoss"] = []
    LossDict["valid/4th/fape40_2/realCadistLoss"] = []

    LossDict["valid/4th/fape40_2/plddt_loss"] = []

    LossDict["valid/4th/fape40_2/truelddtmax"] = []
    LossDict["valid/4th/fape40_2/truelddtmean"] = []
    LossDict["valid/4th/fape40_2/truelddtmedian"] = []

    LossDict["valid/4th/fape40_2/plddtmax"] = []
    LossDict["valid/4th/fape40_2/plddtmean"] = []
    LossDict["valid/4th/fape40_2/plddtmedian"] = []

    return LossDict
