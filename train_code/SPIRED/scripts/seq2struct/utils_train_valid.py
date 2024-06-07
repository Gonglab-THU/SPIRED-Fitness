import torch
import numpy as np
import os, time
import scipy
import pandas as pd
import Loss


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


def return_predcadist_plddt(Predxyz, Plddt):
    eps = 1e-8
    xyz_LL3 = Predxyz["4th"][-1]  # (batch=1,3,L,L)
    batch, _, L, _ = xyz_LL3.shape

    xyz_LLL3 = xyz_LL3[:, :, :, None, :] - xyz_LL3[:, :, :, :, None]  # (N,3,L,L,L)
    predcadist = torch.sqrt(torch.sum(xyz_LLL3 * xyz_LLL3, dim=1) + eps)  # (N,L,L,L)

    plddt_tensor = Plddt["4th"][-1]  # plddt_tensor=(1,L,L)
    plddt_tensor = torch.mean(torch.mean(plddt_tensor, dim=0), dim=1)

    window = L // 18

    plddt_16 = list(torch.split(plddt_tensor, window))[1:17]
    plddt_16 = torch.stack(plddt_16)  # (16, window)
    plddt_16max = torch.max(plddt_16, dim=-1)
    index_1 = plddt_16max[1]
    plddt_16max_index = index_1 + torch.arange(window, window * 17, window).to(device=index_1.device)

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


def getDataTrain(domainbatch, PDB_coords_seq_hdf5, Lmax, model_CPU, model_GPU, batch_converter_esm2, device_esm, device1, device2, FLAGS):

    Length = []
    featuredict = {}
    labeldict = {}

    s0 = time.time()

    for domain in domainbatch:
        target_tokens = PDB_coords_seq_hdf5[domain]["target_tokens"][:]  # (1,L)
        Length.append(target_tokens.shape[-1])

        featuredict[domain] = target_tokens
        xyz = PDB_coords_seq_hdf5[domain]["xyz"][:]  # (L,4,3) [N, CA, C, CB]
        psi_phi = PDB_coords_seq_hdf5[domain]["Psi_Phi"][:]  # (L-1, 2) [psi, phi]

        # labelall = torch.from_numpy(PDB_coords_seq_hdf5[domain]['labelall'][:]) # (8, L, L)
        # labeldict[domain] = [xyz, psi_phi, labelall]
        labeldict[domain] = [xyz, psi_phi]

    Lmin = np.min(Length)
    if Lmin <= Lmax:
        for i in range(len(Length)):
            domain = domainbatch[i]
            target_tokens = featuredict[domain]
            # xyz, psi_phi, labelall = labeldict[domain]
            xyz, psi_phi = labeldict[domain]

            if Length[i] > Lmin:
                L = Length[i]
                beg = np.random.randint(L - Lmin + 1)  # random int in [0,L-Lmax+1)
                target_tokens = target_tokens[:, beg : beg + Lmin]
                xyz = xyz[beg : beg + Lmin, :, :]
                psi_phi = psi_phi[beg : beg + Lmin - 1, :]
                # labelall = labelall[:,beg:beg+Lmin,beg:beg+Lmin]

            featuredict[domain] = target_tokens
            labelall = comp_feature(xyz)  # (8,L,L)
            labeldict[domain] = [labelall, psi_phi, xyz]
    else:
        for domain in domainbatch:
            target_tokens = featuredict[domain]
            # xyz, psi_phi, labelall = labeldict[domain]
            xyz, psi_phi = labeldict[domain]

            L = target_tokens.shape[-1]
            beg = np.random.randint(L - Lmax + 1)  # random int in [0,L-Lmax+1)

            target_tokens = target_tokens[:, beg : beg + Lmax]
            xyz = xyz[beg : beg + Lmax, :, :]
            psi_phi = psi_phi[beg : beg + Lmax - 1, :]
            # labelall = labelall[:, beg:beg+Lmax, beg:beg+Lmax]

            featuredict[domain] = target_tokens
            labelall = comp_feature(xyz)  # (8,L,L)

            labeldict[domain] = [labelall, psi_phi, xyz]

    token_list, label_list, psi_phi_list, xyz_list = ([], [], [], [])
    for domain in domainbatch:
        token_list.append(featuredict[domain])
        label_list.append(labeldict[domain][0].unsqueeze(0))  # (1,8,L,L)
        psi_phi_list.append(torch.tensor(labeldict[domain][1]).permute(1, 0).unsqueeze(0))  # (1,2,L)
        xyz_list.append(torch.tensor(labeldict[domain][2]).unsqueeze(0))  # (1,L,4,3)

    labelbatch = torch.cat(label_list, dim=0).to(device2)  # (batch, 8, L, L)
    psi_phi_batch = torch.cat(psi_phi_list, dim=0).to(device2)  # (batch, 2, L)
    xyz_batch = torch.cat(xyz_list, dim=0).to(device2)  # (batch, L,4,3)

    s1 = time.time()
    t_label = s1 - s0
    print("t_label", round(t_label, 2))

    target_tokens = np.concatenate(token_list, axis=0)  # (batch, L )
    seq_str = tokens2seq(target_tokens)
    labels, strs, target_tokens = batch_converter_esm2(seq_str)  # target_tokens = (batch, L+2 )
    f1dbatch, f2dbatch = ESM2_embed(target_tokens, batch_converter_esm2, model_CPU, model_GPU, length_cutoff=FLAGS.ESM_length_cutoff, gpu=device_esm)  # f1dbatch=(N,L,C), f2dbatch=(N,C,L,L)
    f1dbatch = f1dbatch.to(device1)
    s2 = time.time()
    t_esm2 = s2 - s1
    print("t_esm2", round(t_esm2, 2))
    return (labelbatch, psi_phi_batch, f1dbatch, f2dbatch, target_tokens[:, 1:-1], xyz_batch)


def getDataValid(domain, PDB_coords_seq_hdf5, model_CPU, model_GPU, batch_converter_esm2, device_esm, device1, device2, FLAGS):

    s0 = time.time()
    target_tokens = PDB_coords_seq_hdf5[domain]["target_tokens"][:]  # (1,L)

    xyz = PDB_coords_seq_hdf5[domain]["xyz"][:]  # (L,4,3) [N, CA, C, CB]
    psi_phi = PDB_coords_seq_hdf5[domain]["Psi_Phi"][:]  # (L-1, 2) [cos(psi), cos(phi)]
    # labelall = torch.from_numpy(label[domain]['labelall'][:]) # (8, L, L)
    L = target_tokens.shape[-1]

    if L > FLAGS.valid_length:
        target_tokens = target_tokens[:, : FLAGS.valid_length]
        xyz = xyz[: FLAGS.valid_length, :, :]
        psi_phi = psi_phi[: FLAGS.valid_length - 1, :]
        # labelall = labelall[:, :FLAGS.valid_length, :FLAGS.valid_length]

    labelall = comp_feature(xyz)  # (8,L,L)
    seq_str = tokens2seq(target_tokens)
    labels, strs, target_tokens = batch_converter_esm2(seq_str)
    f1d, f2d = ESM2_embed(target_tokens, batch_converter_esm2, model_CPU, model_GPU, length_cutoff=FLAGS.ESM_length_cutoff, gpu=device_esm)  # f1dbatch=(N,L,C), f2dbatch=(N,C,L,L)

    f1d = f1d.to(device1)
    s1 = time.time()
    t_esm = s1 - s0
    print("t_esm", round(t_esm, 2))

    labelall = labelall.to(device2).unsqueeze(0)  # (1, 8, L, L)
    psi_phi = torch.tensor(psi_phi).permute(1, 0).unsqueeze(0).to(device2)  # (1, 2, L)
    xyz = torch.tensor(xyz).unsqueeze(0).to(device2)  # (1,L,4,2)

    s2 = time.time()
    t_label = s2 - s1
    print("t_label", round(t_label, 2))

    return (labelall, psi_phi, f1d, f2d, target_tokens[:, 1:-1], seq_str[0][1], xyz)


def threadTrain(model, target_tokens, f1dbatch, f2dbatch, labelbatch, label_phi_psi, FLAGS, sample_idx):
    # f1dbatch = (batch, L, 37, 2560); f2dbatch = (batch, channel, L, L); labelbatch=(batch,8,L,L)

    batch, channel, L, _ = f2dbatch.shape
    start_idx = sample_idx % 2
    coord_idx = list(range(start_idx, L, 2))

    if FLAGS.train_cycle > 1:
        cycle = np.random.randint(low=1, high=FLAGS.train_cycle)
    else:
        cycle = 1

    print("cycle number:", cycle)
    s0 = time.time()
    Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, f1d_cycle, f2d_cycle = model(target_tokens, f1dbatch, no_recycles=cycle)
    s1 = time.time()

    Losse2e, lossRD_byResidue = Loss.getLossMultBlock(Predxyz, PredCadistavg, Plddt, labelbatch, coord_idx)
    LossCE = Loss.getLossCE(cb, omega, theta, phi, labelbatch)
    LossCA = Loss.getLossCAL1(ca, labelbatch, cutoff=32)
    loss_phi_psi_1D = Loss.getLoss_phi_psi_1D_L1(phi_psi_1D, label_phi_psi)
    # Lossoptim = np.sqrt(L) * (4 * Losse2e['optim'] + 0.1 * LossCE['optim'] + 0.5* loss_phi_psi_1D)
    Lossoptim = np.sqrt(L) * (4 * Losse2e["optim"] + 0.01 * LossCE["optim"] + 0.5 * loss_phi_psi_1D)

    # print("\n", "Loss_SUM", Lossoptim.item())
    # print("LossRD_SUM", np.sqrt(L) *4 * Losse2e['optim'].item())
    # print("LossCE", np.sqrt(L) *0.01 * LossCE['optim'].item())
    # print("loss_phi_psi_1D", np.sqrt(L) *0.5* loss_phi_psi_1D.item(),"\n")

    s2 = time.time()
    t_train1 = s1 - s0
    t_train2 = s2 - s1
    # print("threadTrain", "train time", round(t_train1,2), "loss time", round(t_train2,2))

    return (Predxyz, PredCadistavg, Plddt, phi_psi_1D, Lossoptim, Losse2e, LossCE, LossCA, loss_phi_psi_1D)


def threadValid(sample, model, target_tokens, f1dbatch, f2dbatch, labelbatch, label_phi_psi, FLAGS, sample_idx):
    # labelbatch = (batch, 8, L, L)

    batch, channel, L, _ = f2dbatch.shape
    cycle = FLAGS.valid_cycle
    start_idx = sample_idx % 2
    print("sample_idx", sample_idx)
    coord_idx = list(range(start_idx, L, 2))

    s0 = time.time()
    Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, f1d_cycle, f2d_cycle = model(target_tokens, f1dbatch, no_recycles=cycle)
    s1 = time.time()

    Losse2e, lossRD_byResidue = Loss.getLossMultBlock(Predxyz, PredCadistavg, Plddt, labelbatch, coord_idx)
    LossCE = Loss.getLossCE(cb, omega, theta, phi, labelbatch)
    LossCA = Loss.getLossCAL1(ca, labelbatch, cutoff=32)
    loss_phi_psi_1D = Loss.getLoss_phi_psi_1D_L1(phi_psi_1D, label_phi_psi)

    # Lossoptim = np.sqrt(L) * (4 * Losse2e['optim'] + 0.1 * LossCE['optim'] + 0.5 * loss_phi_psi_1D)
    Lossoptim = np.sqrt(L) * (4 * Losse2e["optim"] + 0.01 * LossCE["optim"] + 0.5 * loss_phi_psi_1D)

    # print("\n", "Lossoptim", Lossoptim.item())
    # print("Losse2e['optim']", np.sqrt(L) * 4 * Losse2e['optim'].item())
    # print("LossCE['optim']", np.sqrt(L) * 0.01 * LossCE['optim'].item())
    # print("loss_phi_psi_1D", np.sqrt(L) * 0.5 * loss_phi_psi_1D.item(), "\n")

    s2 = time.time()
    t_train1 = s1 - s0
    t_train2 = s2 - s1
    # print("threadTrain", "train_time", round(t_train1,2), "loss_time", round(t_train2,2))

    return (Predxyz, PredCadistavg, Plddt, phi_psi_1D, Lossoptim, Losse2e, LossCE, LossCA, loss_phi_psi_1D)


def threadTest(domain, model, model_cpu, hdf5, esm2_CPU, esm2_GPU, batch_converter_esm2, device_esm, device1, FLAGS):

    target_tokens_0 = hdf5[domain]["target_tokens"][:]  # (1,L)
    seq_str = tokens2seq(target_tokens_0)

    L = target_tokens_0.shape[-1]
    labels, strs, target_tokens = batch_converter_esm2(seq_str)

    with torch.no_grad():
        if L <= FLAGS.maxlen:
            f1d, f2d = ESM2_embed(target_tokens, batch_converter_esm2, esm2_CPU, esm2_GPU, length_cutoff=FLAGS.ESM_length_cutoff, gpu=device_esm)
            f1d = f1d.to(device1)
            Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, f1d_cycle, f2d_cycle = model(target_tokens[:, 1:-1], f1d, no_recycles=FLAGS.fold_cycles)
        else:
            f1d, f2d = ESM2_embed(target_tokens, batch_converter_esm2, esm2_CPU, esm2_GPU, length_cutoff=FLAGS.ESM_length_cutoff, gpu="cpu")
            target_tokens, f1d = (target_tokens.to(device="cpu"), f1d.to(device="cpu"))
            Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, f1d_cycle, f2d_cycle = model_cpu(target_tokens[:, 1:-1], f1d, no_recycles=FLAGS.fold_cycles)

    return (Predxyz, PredCadistavg, Plddt, phi_psi_1D, f1d_cycle, f2d_cycle, seq_str[0][1], target_tokens_0)


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


def write_pdb(xyz, target_seq, coords_num, name, dir):
    dir = os.path.join(dir, "pdb", name)
    if not os.path.exists(dir):
        os.system("mkdir -p " + dir)
    # xyz = (N,L,3), N is the number of coordinate sets
    N, L, _ = xyz.shape
    print("out xyz", xyz.shape)
    if coords_num > N:
        coords_num = N
    aa_dict = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL", "X": "ALA"}
    for n in range(coords_num):
        path = os.path.join(dir, name + "_" + str(n + 1) + ".pdb")
        xyz_L = xyz[n, ...]  # (L, 3)
        with open(path, "w") as FILE:
            for i in range(L):
                amino_acid = aa_dict[target_seq[i]]
                xyz_ca = xyz_L[i, ...]  # (3,)
                (x, y, z) = (round(float(xyz_ca[0]), 3), round(float(xyz_ca[1]), 3), round(float(xyz_ca[2]), 3))
                FILE.write("ATOM  {:>5} {:<4} {} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2}  \n".format(int(i + 1), "CA", amino_acid, int(i + 1), x, y, z, 1.0, 0.0, "C"))


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


def makelog(logname=None, log=[], commandline=None, header=None):
    if not os.path.exists(logname):
        with open(logname, "w") as fout:
            for d in range(len(commandline) - 1):
                fout.write(str(commandline[d]) + "\t")
            fout.write(str(commandline[-1]) + "\n")
            for d in range(len(header) - 1):
                fout.write(str(header[d]) + "\t")
            fout.write(str(header[-1]) + "\n")
            for d in range(len(log) - 1):
                fout.write(str(log[d]) + "\t")
            fout.write(str(log[-1]) + "\n")
    else:
        with open(logname, "a+") as fout:
            for d in range(len(log) - 1):
                fout.write(str(log[d]) + "\t")
            fout.write(str(log[-1]) + "\n")


## load the saved model parameters
def load_model_parameters(model, saved_model_path):
    if saved_model_path == "None":
        print("Use initiated model...")
        return model
    else:
        print("Loading saved model: ", saved_model_path)
        init_model_dict = model.state_dict()
        saved_model_dict = torch.load(saved_model_path, map_location="cpu")

        for k, v in saved_model_dict.items():
            if k in init_model_dict.keys():
                init_model_dict[k] = v

        model.load_state_dict(init_model_dict)
        return model


# write loss data into xls format
def write_loss_table(data_dict, output_dir, file_name):
    Loss_df = pd.DataFrame.from_dict(data_dict)
    Loss_table_path = os.path.join(output_dir, file_name)
    Loss_df.to_csv(Loss_table_path, sep="\t", na_rep="nan", index=False)