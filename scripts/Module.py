import torch
from torch import nn

eps = 1e-8


class IRCN(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.gammaRow = torch.nn.Parameter(torch.ones((1, channel, 1, 1)))
        self.betaRow = torch.nn.Parameter(torch.zeros((1, channel, 1, 1)))

        self.gammaCol = torch.nn.Parameter(torch.ones((1, channel, 1, 1)))
        self.betaCol = torch.nn.Parameter(torch.zeros((1, channel, 1, 1)))

        self.gammaIns = torch.nn.Parameter(torch.zeros((1, channel, 1, 1)))
        self.betaIns = torch.nn.Parameter(torch.zeros((1, channel, 1, 1)))

    def forward(self, inputs):
        Batch, channels, heights, widths = inputs.shape

        # row normalization
        means = torch.mean(inputs, dim=3, keepdim=True)  # (Batch, channels, heights, 1)
        var = torch.var(inputs, dim=3, unbiased=False, keepdim=True)  # (Batch, channels, heights, 1)
        normRow = self.gammaRow * (inputs - means) / (torch.sqrt(var + eps) + eps) + self.betaRow

        # column normalization
        means = torch.mean(inputs, dim=2, keepdim=True)  # (Batch, channels, 1, widths)
        var = torch.var(inputs, dim=2, unbiased=False, keepdim=True)  # (Batch, channels, 1, widths)
        normCol = self.gammaCol * (inputs - means) / (torch.sqrt(var + eps) + eps) + self.betaCol

        # instance normalization
        means = torch.mean(inputs, dim=(2, 3), keepdim=True)  # (Batch, channels, 1, 1)
        var = torch.var(inputs, dim=(2, 3), unbiased=False, keepdim=True)  # (Batch, channels, 1, 1)
        normIns = self.gammaIns * (inputs - means) / (torch.sqrt(var + eps) + eps) + self.betaIns

        outputs = torch.cat([normIns, normRow, normCol], dim=1)
        return outputs


class predxyz(nn.Module):
    def __init__(self, channel):
        super().__init__()

        ### https://arxiv.org/pdf/1603.05027.pdf

        self.conv = nn.Conv2d(3 * channel, channel, kernel_size=(1, 1))
        self.cnn = nn.Sequential(nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0.2), nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(3 - 1) // 2), nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0.2), nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(3 - 1) // 2))
        self.nad = nn.Sequential(nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0.2))
        self.toxyz = nn.Conv2d(channel, 3, kernel_size=(1, 1))  # no activation

    def forward(self, out):
        out = self.conv(out)
        out = self.cnn(out) + out
        out = self.nad(out)
        predxyz = self.toxyz(out)

        return predxyz, out


class predDxyz(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.ins = nn.InstanceNorm2d(channel + 13, affine=True)
        self.conv = nn.Conv2d(channel + 13, channel, kernel_size=(1, 1))

        self.block = nn.Sequential(nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0.2), nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(3 - 1) // 2), nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0.2), nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(3 - 1) // 2))
        self.nad = nn.Sequential(nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0.2))
        self.todxyz = nn.Conv2d(channel, 3, kernel_size=(1, 1))

    def forward(self, featNCLL, cadistavg):
        ### cadistavg.shape (N,L,L)
        cutoff = torch.tensor([8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]).reshape(13, 1, 1).to(cadistavg.device)
        cadistavg = torch.tile(cadistavg[:, None, :, :], [1, 13, 1, 1])
        out = (cadistavg <= cutoff).type(torch.float)  # (N,13,L,L)
        out = torch.cat([featNCLL, out], dim=1)  # (N,channel+13,L,L)

        out = self.ins(out)
        out = self.conv(out)
        out = self.block(out) + out  # resnet short cut
        out = self.nad(out)
        xyz = self.todxyz(out)

        return xyz, out


class predDxyz_shareWeight(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.ins = nn.InstanceNorm2d(channel + 13, affine=True)
        self.conv_cadist = nn.Conv2d(channel + 13, channel, kernel_size=(1, 1))

        self.block = nn.Sequential(nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0), nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(3 - 1) // 2), nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0), nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(3 - 1) // 2))
        self.nad = nn.Sequential(nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0))
        self.todxyz = nn.Conv2d(channel, 3, kernel_size=(1, 1))

    def forward(self, featNCLL, cadistavg=None):

        featNCLL = featNCLL.permute(0, 3, 1, 2).contiguous()
        if cadistavg is not None:
            ### cadistavg.shape (N,L,L)  featNCLL.shape (N,channel,L,L)
            cutoff = torch.tensor([8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]).reshape(13, 1, 1).to(cadistavg.device)
            cadistavg = torch.tile(cadistavg[:, None, :, :], [1, 13, 1, 1])
            out = (cadistavg <= cutoff).type(torch.float)  # (N,13,L,L)
            out = torch.cat([featNCLL, out], dim=1)  # (N,channel+13,L,L)
            out = self.ins(out)
            out = self.conv_cadist(out)
        else:
            out = featNCLL

        out = self.block(out) + out  # resnet short cut
        out = self.nad(out)
        xyz = self.todxyz(out)

        out = out.permute(0, 2, 3, 1).contiguous()
        return xyz, out


class pred_CB_Angle(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.ins = nn.InstanceNorm2d(channel + 13, affine=True)
        self.conv = nn.Conv2d(channel + 13, channel, kernel_size=(1, 1))

        self.block = nn.Sequential(nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0), nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(3 - 1) // 2), nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0), nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(3 - 1) // 2))
        self.nad = nn.Sequential(nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0))

        self.cbConv = nn.Conv2d(channel, 48, kernel_size=(1, 1))
        self.omegaConv = nn.Conv2d(channel, 25, kernel_size=(1, 1))
        self.thetaConv = nn.Conv2d(channel, 25, kernel_size=(1, 1))
        self.phiConv = nn.Conv2d(channel, 13, kernel_size=(1, 1))

    def forward(self, featNCLL, cadistavg):
        ### cadistavg.shape (N,L,L)
        featNCLL = featNCLL.permute(0, 3, 1, 2).contiguous()
        cutoff = torch.tensor([8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]).reshape(13, 1, 1).to(cadistavg.device)
        cadistavg = torch.tile(cadistavg[:, None, :, :], [1, 13, 1, 1])
        out = (cadistavg <= cutoff).type(torch.float)  # (N,13,L,L)
        out = torch.cat([featNCLL, out], dim=1)  # (N,channel+13,L,L)

        out = self.ins(out)
        out = self.conv(out)
        out = self.block(out) + out
        out = self.nad(out)

        cb = self.cbConv((out + out.permute(0, 1, 3, 2)) / 2.0)
        omega = self.omegaConv((out + out.permute(0, 1, 3, 2)) / 2.0)
        theta = self.thetaConv(out)
        phi = self.phiConv(out)

        return cb, omega, theta, phi  # (N,C,L,L)


class to_CA_CB_Angle(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.caConv = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.cbConv = nn.Conv2d(channel, 48, kernel_size=(1, 1))
        self.omegaConv = nn.Conv2d(channel, 25, kernel_size=(1, 1))
        self.thetaConv = nn.Conv2d(channel, 25, kernel_size=(1, 1))
        self.phiConv = nn.Conv2d(channel, 13, kernel_size=(1, 1))

    def forward(self, out):
        out = out.permute(0, 3, 1, 2).contiguous()
        ca = self.caConv((out + out.permute(0, 1, 3, 2)) / 2.0)
        cb = self.cbConv((out + out.permute(0, 1, 3, 2)) / 2.0)
        omega = self.omegaConv((out + out.permute(0, 1, 3, 2)) / 2.0)
        theta = self.thetaConv(out)
        phi = self.phiConv(out)

        return ca, cb, omega, theta, phi  # (N,C,L,L)


class to_phi_psi(nn.Module):
    def __init__(self, channel=1024):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 1, kernel_size=(15, 15), stride=1, padding=7)
        self.elu = nn.ELU()
        self.conv_2 = nn.Conv2d(1, 2, kernel_size=(channel, 2), stride=1, padding=0)

    def forward(self, f1d):
        # f1d = (N, L, 1024)
        out = f1d.permute(0, 2, 1).unsqueeze(1).contiguous()  # (N, 1, 1024, L)
        out = self.conv_1(out)  # (N, 1, 1024, L)
        out = self.elu(out)
        out = self.conv_2(out)  # (N, 2, 1, L-1)
        out = out.squeeze(2).contiguous()
        return out  # (N, 2, L-1)


class forNextBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.ins = nn.InstanceNorm2d(channel + 13, affine=True)
        self.conv = nn.Conv2d(channel + 13, channel, kernel_size=(1, 1))

        self.block = nn.Sequential(nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0.1), nn.Conv2d(channel, channel, kernel_size=(1, 1)), nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0.1), nn.Conv2d(channel, channel, kernel_size=(1, 1)))
        self.nad = nn.Sequential(nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0))

    def forward(self, featNCLL, cadistavg):
        ### cadistavg.shape (N,L,L)
        featNCLL = featNCLL.permute(0, 3, 1, 2).contiguous()
        cutoff = torch.tensor([8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]).reshape(13, 1, 1).to(cadistavg.device)
        cadistavg = torch.tile(cadistavg[:, None, :, :], [1, 13, 1, 1])
        out = (cadistavg <= cutoff).type(torch.float)  # (N,13,L,L)
        out = torch.cat([featNCLL, out], dim=1)  # (N,channel+13,L,L)

        out = self.ins(out)
        out = self.conv(out)
        out = self.block(out) + out
        out = self.nad(out)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out  # (N,C,L,L)


class pLDDT(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.ins = nn.InstanceNorm2d(channel + 13, affine=True)
        self.conv = nn.Conv2d(channel + 13, channel, kernel_size=(1, 1))

        self.block = nn.Sequential(nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0.1), nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(3 - 1) // 2), nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0.1), nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(3 - 1) // 2))
        self.nad = nn.Sequential(nn.InstanceNorm2d(channel, affine=True), nn.ELU(), nn.Dropout(p=0.1))

        self.plddt = nn.Conv2d(channel, 1, kernel_size=(1, 1))

    def forward(self, featNCLL, cadistavg):
        ### cadistavg.shape (N,L,L)
        featNCLL = featNCLL.permute(0, 3, 1, 2).contiguous()
        cutoff = torch.tensor([8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]).reshape(13, 1, 1).to(cadistavg.device)
        cadistavg = torch.tile(cadistavg[:, None, :, :], [1, 13, 1, 1])
        out = (cadistavg <= cutoff).type(torch.float)  # (N,13,L,L)
        out = torch.cat([featNCLL, out], dim=1)  # (N,channel+13,L,L)

        out = self.ins(out)
        out = self.conv(out)
        out = self.block(out) + out
        out = self.nad(out)

        plddt = torch.sigmoid(self.plddt(out))

        return plddt[:, 0, :, :]  # (N,L,L)
