import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    """
    Registers the coordinates of CA atoms and the rotations of local coordinate 
    systems as learnable parameters.

    batch (int): number of structures to predict simultaneously
    length (int): length of the target protein sequence
    """
    def __init__(self, batch, length):
        super(Network, self).__init__()
        self.CA = nn.Parameter(torch.rand(batch, length, 3))
        self.CA.data.uniform_(-1, 1)
        self.Theta = nn.Parameter(torch.rand(batch, length, 3))
        self.Theta.data.uniform_(-math.pi, math.pi)

    def forward(self, x):
        CA = self.CA
        Theta = self.Theta
        return CA, Theta


class GradientDescent:
    """"
    Protein folding environment based on gradient descent.

    Takes protein sequence and predicted geometric information as input and constructs multiple
    constraints loss functions to update the network parameters and returns the coordinates
    of backbone atoms.

    seq (list): protein sequence in numerical coding
    pred (dict): predicted geometric information
    params (dict): statistical parameters
    npose (int): number of structures to predict simultaneously
    decive (str): if 'cuda:0', run on GPU 0, else on cpu
    steps (int): number of optimization steps
    """
    def __init__(self, seq, pred, params, npose=1, steps=400, device='cpu'):
        self.seq = seq
        self.length = len(seq)
        self.pred = pred
        self.params = params
        self.npose = npose
        self.steps = steps
        self.device = device
        self.epsilon = 1e-6
        self.model = Network(npose, self.length).to(device)
        self._reshape()

    def _reshape(self):
        self.local = self.params['local'][self.seq].repeat(self.npose, 1, 1, 1).to(self.device)

    def _theta2mat(self, theta):
        '''
        Convert Euler Angle into rotation matrix.

        theta (tensor): Euler Angle, shape = [batch, length, 3]
        '''
        zeron = torch.zeros(self.npose, self.length, device=self.device)
        onen = torch.ones(self.npose, self.length, device=self.device)

        sin1 = torch.sin(theta[..., 0])
        cos1 = torch.cos(theta[..., 0])
        sin2 = torch.sin(theta[..., 1])
        cos2 = torch.cos(theta[..., 1])
        sin3 = torch.sin(theta[..., 2])
        cos3 = torch.cos(theta[..., 2])

        mat_x = torch.stack([onen, zeron, zeron,
                            zeron, cos1, -sin1,
                            zeron, sin1, cos1], dim=-1).view(
                            self.npose, self.length, 3, 3) 
        
        mat_y = torch.stack([cos2, zeron, sin2, 
                            zeron, onen, zeron, 
                            -sin2, zeron, cos2], dim=-1).view(
                            self.npose, self.length, 3, 3)
        
        mat_z = torch.stack([cos3, -sin3, zeron, 
                            sin3, cos3, zeron, 
                            zeron, zeron, onen], dim=-1).view(
                            self.npose, self.length, 3, 3)
        Mat = mat_z @ mat_y @ mat_x
        return Mat
    
    def _coords2rot(self, a, b, c):
        '''
        Calculate the rotation of local coordinate system.

        a, b, c (tensor): global coordinates, shape = [batch, length, 3]
        '''
        axis_x = F.normalize(a - b)
        axis_z = F.normalize(torch.cross(axis_x, c - b))
        axis_y = torch.cross(axis_z, axis_x)
        axis = torch.stack([axis_x, axis_y, axis_z], dim=-2)
        U, _, VH = torch.linalg.svd(axis)
        return torch.inverse(VH) @ torch.inverse(U)
    
    def _oxygen(self, mat, coords):
        '''
        Infer the coordinates of oxygen atoms.

        mat (tensor): rotation matrix, shape = [batch, length, 3, 3]
        coords (list): global coordinates of the backbone atoms
        '''
        oxy = self.params['oxygen'].to(self.device)
        oxy_last = self.params['oxygen_last'].to(self.device)
        a, b, c = coords[2][:, 1:], coords[1][:, :-1], coords[0][:, :-1]
        plane = self._coords2rot(a, b, c)
        oxygen = plane @ oxy + b
        oxygen_last = coords[0][:, -1] + torch.squeeze(mat[:, -1] @ (self.local[:, -1, :, 1] + 
                                                                     oxy_last).unsqueeze(-1))
        return torch.cat([oxygen, oxygen_last.unsqueeze(1)], dim=1)
    
    def _hydrogen(self, mat, coords):
        '''
        Infer the coordinates of hydrogen atoms.

        mat (tensor): rotation matrix, shape = [batch, length, 3, 3]
        coords (list): global coordinates of the backbone atoms
        '''
        hyd = self.params['hydrogen'].to(self.device)
        hyd_first = self.params['hydrogen_first'].to(self.device)
        a, b, c = coords[1][:, :-1], coords[2][:, 1:], coords[0][:, 1:]
        plane = self._coords2rot(a, b, c)
        hydrogen = plane @ hyd + b
        hydrogen_first = coords[0][:, 0] + torch.squeeze(mat[:, 0] @ (self.local[:, 0, :, 2] + 
                                                                      hyd_first).unsqueeze(-1))
        return torch.cat([hydrogen_first.unsqueeze(1), hydrogen], dim=1)
    
    def _info(self, CA, Theta, OH=False):
        '''
        Calculate the global coordinates of backbone atoms.
        
        CA (tensor): coordinates of CA atoms, shape = [batch, length, 3]
        Theta (tensor): rotations of local coordinate systems, shape = [batch, length, 3]
        OH (bool): if true, calculate the coordinates of oxygen and hydrogen atoms
        '''
        mat = self._theta2mat(Theta)
        transpose = mat @ self.local
        coords = [transpose[..., i] + CA for i in range(5)]
        if OH:
            O = self._oxygen(mat, coords)
            H = self._hydrogen(mat, coords)
            coords.extend((O, H))
        return mat, coords
    
    def _peptide_term(self, coords):
        '''
        Construct the peptide plane constraints
        '''
        bond_label = self.params['peptide_bond'].to(self.device)
        angleA_label = self.params['angleA'].to(self.device)
        angleB_label = self.params['angleB'].to(self.device)
        omega_label = self.params['omega'].to(self.device)
        
        # peptide bond constraint
        bond_pred = torch.linalg.norm(coords[1][:, :-1] - coords[2][:, 1:], dim=-1)
        forward = F.pad(torch.abs(bond_pred - bond_label), [0, 1])
        backward = F.pad(torch.abs(bond_pred - bond_label), [1, 0])
        bond_loss = torch.sum(forward + backward, dim=-1).mean()

        # peptide angle CA(i-1)-C(i-1)-N(i) constraint
        c_ca = F.normalize(coords[0][:, :-1] - coords[1][:, :-1], dim=-1)
        c_n = F.normalize(coords[2][:, 1:] - coords[1][:, :-1], dim=-1)
        angleA = torch.acos(torch.sum(c_ca * c_n, dim=-1))
        angleA_loss = torch.sum(torch.abs(angleA - angleA_label), dim=-1).mean()

        # peptide angle C(i-1)-N(i)-CA(i) constraint
        n_c = F.normalize(coords[1][:, :-1] - coords[2][:, 1:], dim=-1)
        n_ca = F.normalize(coords[0][:, 1:] - coords[2][:, 1:], dim=-1)
        angleB = torch.acos(torch.sum(n_c * n_ca, dim=-1))
        angleB_loss = torch.sum(torch.abs(angleB - angleB_label), dim=-1).mean()

        # peptide plane angle constraint
        pro_mask = torch.ones(self.length - 1, device=self.device)
        pro_index = [i for i,x in enumerate(self.seq[1:]) if x == 13]
        pro_mask[pro_index] = 0.0
        pro_mask = pro_mask.repeat(self.npose, 1)
        v1 = torch.cross((coords[0][:, :-1] - coords[1][:, :-1]), 
                         (coords[2][:, 1:] - coords[1][:, :-1]))
        v2 = torch.cross((coords[1][:, :-1] - coords[2][:, 1:]),
                         (coords[0][:, 1:] - coords[2][:, 1:]))
        omega_pred = torch.sum(F.normalize(v1, dim=-1) * F.normalize(v2, dim=-1), dim=-1)
        omega_loss = pro_mask * torch.abs(omega_pred - omega_label)
        mask_rev = torch.ones_like(pro_mask, device=self.device) - pro_mask
        pro_loss = mask_rev * torch.abs(torch.abs(omega_pred) - torch.abs(omega_label))
        omega_loss = torch.sum(omega_loss + pro_loss, dim=-1).mean()
        return bond_loss + angleA_loss + angleB_loss + omega_loss
    
    def _global_term(self, coords):
        '''
        Construct the inter-residue distance constraint
        '''
        dist_label = self.params['CA_dist'].to(self.device)
        dist = torch.linalg.norm(coords[0][:, :-1] - coords[0][:, 1:], dim=-1)
        forward = F.pad(torch.abs(dist - dist_label), [0, 1])
        backward = F.pad(torch.abs(dist - dist_label), [1, 0])
        global_term = torch.sum(forward + backward, dim=-1)
        return global_term.mean()
    
    def _vdw_term(self, coords):
        '''
        Construct the Van der Waals repulsive force constraint
        '''
        vdw_mask = self.params['vdw_mask'].repeat(self.npose, 1, 1).to(self.device)
        vdw_dist = self.params['vdw_dist'].repeat(self.npose, 1, 1).to(self.device)
        all_coords = torch.stack(coords, dim=-2).view(self.npose, -1, 3)
        dist = torch.linalg.norm(all_coords.unsqueeze(1) - all_coords.unsqueeze(2), dim=-1)
        vdw_loss = torch.sum(torch.clamp(vdw_mask * (vdw_dist - dist), min=0.0), dim=[1, 2])
        return vdw_loss.mean()
    
    def _vector_term(self, mat, coords):
        '''
        Construct the predicted constraint
        '''
        loss = nn.MSELoss(reduction='none')
        reference = self.pred['reference']
        rotation = self.pred['rotation'].repeat(self.npose, 1, 1, 1, 1).to(self.device)
        translation = self.pred['translation'].repeat(self.npose, 1, 1, 1).to(self.device)
        local = self.local[..., :4].unsqueeze(1)
        vectors_label = rotation @ local + translation.unsqueeze(-1)
        backbone = torch.stack(coords[:4], dim=-1).unsqueeze(1)
        CA_ref = coords[0][:, reference].unsqueeze(2).unsqueeze(-1)
        vectors_pred = torch.inverse(mat[:, reference]).unsqueeze(2) @ (backbone - CA_ref)
        self.vector_loss = torch.sum(loss(vectors_pred, vectors_label), dim=-2).sum(-1)
        return self.vector_loss.mean()
  
    def _dihedral_term(self, mat, coords):
        '''
        Construct the dihedral angle constraint
        '''
        loss = nn.MSELoss(reduction='none')
        dihedrals = self.pred['dihedrals'].to(self.device)
        pep_bond = self.params['peptide_bond'].to(self.device)
        psi, phi = dihedrals[1:, 0], dihedrals[:-1, 1]
        # infer the coordinates of the last C atom
        def last_C(phi):
            CA, C, N = [self.local[0, :, :, i] for i in range(3)]
            bond = torch.linalg.norm(N - CA, dim=-1)
            cos_angle = torch.sum(F.normalize(N - CA, dim=-1) * 
                                  F.normalize(C - CA, dim=-1), dim=-1)
            rad_angle = torch.acos(cos_angle) - 1.5708
            v1 = 0.520 * pep_bond
            v2 = torch.sqrt(torch.square(pep_bond) - torch.square(v1) + self.epsilon)
            x = v2 * torch.cos(phi) * torch.cos(rad_angle[1:-1]) + \
                (bond[1:-1] + v1) * cos_angle[1:-1]
            y = v2 * torch.cos(phi) * torch.sin(rad_angle[1:-1]) + \
                (bond[1:-1] + v1) * torch.cos(rad_angle[1:-1])
            z = -torch.sign(phi) * torch.sqrt(torch.square(v2) - 
                                              torch.square(v2 * torch.cos(phi)) + self.epsilon)
            return torch.stack([x, y, z], dim=-1)
        # infer the coordinates of the next N atom
        def next_N(psi):
            x = 0.449 * pep_bond + self.local[0, 1:-1, 0, 1]
            y = torch.sqrt(torch.square(pep_bond) - torch.square(
                0.449 * pep_bond) + self.epsilon) * torch.cos(psi)
            z = torch.sign(psi) * torch.sqrt(torch.square(pep_bond) - torch.square(
                0.449 * pep_bond) - torch.square(y) + self.epsilon)
            return torch.stack([x, y, z], dim=-1)
        local_C, local_N = last_C(phi), next_N(psi)
        label_C = torch.squeeze(mat[:, 1:-1] @ local_C.unsqueeze(-1)) + coords[0][:, 1:-1]
        label_N = torch.squeeze(mat[:, 1:-1] @ local_N.unsqueeze(-1)) + coords[0][:, 1:-1]
        self.dihedral_loss = torch.sum(loss(coords[1][:, :-2], label_C) + 
                                       loss(coords[2][:, 2:], label_N), dim=-1)
        return self.dihedral_loss.mean()
    
    def _step(self, opt, epoch):
        CA, Theta = self.model(epoch)
        if epoch < self.steps:
            for param_group in opt.param_groups:
                param_group['lr'] = 0.6
            mat, coords = self._info(CA, Theta)
        else:
            for param_group in opt.param_groups:
                param_group['lr'] = 0.02
            mat, coords = self._info(CA, Theta, OH=True)
        vector_loss = self._vector_term(mat, coords)
        dihedral_loss = self._dihedral_term(mat, coords)
        if epoch < self.steps:
            loss = vector_loss + dihedral_loss
        else:
            extra_loss = self._peptide_term(coords) + self._global_term(coords)
            if epoch < self.steps + 400:
                loss = vector_loss + dihedral_loss + extra_loss
            else:
                vdw_loss = self._vdw_term(coords)
                if epoch < self.steps + 600:
                    loss = vector_loss + dihedral_loss + extra_loss + 1e1 * vdw_loss
                else:
                    loss = vector_loss + dihedral_loss + extra_loss + 5 * vdw_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    def _fold(self):
        opt = torch.optim.Adam(self.model.parameters())
        for epoch in range(self.steps + 800):
            self._step(opt, epoch)
        CA, Theta = self.model(epoch)
        _, coords = self._info(CA, Theta, OH=True)
        output = torch.stack(coords, dim=-1).detach()
        if self.device != 'cpu':
            output = output.cpu()
        return output.numpy()
    