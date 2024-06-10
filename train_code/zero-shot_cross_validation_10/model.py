import torch


class PretrainGAT(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim):
        super().__init__()

        self.node_dim = node_dim
        self.n_head = n_head
        self.head_dim = node_dim // n_head

        # * to alpha
        self.query = torch.nn.Linear(node_dim, self.head_dim * n_head, bias=False)
        self.key = torch.nn.Linear(node_dim, self.head_dim * n_head, bias=False)
        self.value = torch.nn.Linear(node_dim, self.head_dim * n_head, bias=False)
        self.pair2alpha = torch.nn.Linear(pair_dim, n_head, bias=False)
        self.conv2dalpha = torch.nn.Sequential(torch.nn.InstanceNorm2d(n_head * 2), torch.nn.Conv2d(n_head * 2, n_head, 3, 1, 1), torch.nn.LeakyReLU())
        self.alpha_mask_plddt = torch.nn.Sequential(torch.nn.InstanceNorm2d(n_head + 1), torch.nn.Conv2d(n_head + 1, n_head, 3, 1, 1), torch.nn.LeakyReLU())

        # output
        self.out_transform = torch.nn.Sequential(torch.nn.LayerNorm(n_head * pair_dim + node_dim), torch.nn.Linear(n_head * pair_dim + node_dim, node_dim * 2), torch.nn.LayerNorm(node_dim * 2), torch.nn.LeakyReLU(), torch.nn.Linear(node_dim * 2, node_dim))
        self.layer_norm = torch.nn.LayerNorm(node_dim)
        self.alpha2pair = torch.nn.Sequential(torch.nn.InstanceNorm2d(n_head + pair_dim), torch.nn.Conv2d(n_head + pair_dim, pair_dim, 3, 1, 1), torch.nn.LeakyReLU())

    @staticmethod
    def _heads(x, n_head, n_ch):

        # x = [..., n_head * n_ch] -> [..., n_head, n_ch]
        s = list(x.shape)[:-1] + [n_head, n_ch]
        return x.view(*s)

    def _node2alpha(self, x):
        query_l = self._heads(self.query(x), self.n_head, self.head_dim)
        key_l = self._heads(self.key(x), self.n_head, self.head_dim)

        # query_l = [N, L, n_head, head_dim]
        # key_l = [N, L, n_head, head_dim]

        query_l = query_l.permute(0, 2, 1, 3)
        key_l = key_l.permute(0, 2, 3, 1)

        # query_l = [N, n_head, L, head_dim]
        # key_l = [N, n_head, head_dim, L]

        alpha = torch.matmul(query_l, key_l) / torch.sqrt(torch.FloatTensor([self.head_dim]).to(x.device))
        alpha = alpha.permute(0, 2, 3, 1)
        return alpha

    def _pair2alpha(self, z):
        alpha = self.pair2alpha(z)
        return alpha

    def _node_aggregation(self, alpha, x):
        N = x.shape[0]
        value_l = self._heads(self.value(x), self.n_head, self.head_dim)

        # value_l = [N, L, n_head, head_dim]

        value_l = value_l.permute(0, 2, 1, 3)

        # value_l = [N, n_head, L, head_dim]

        x = torch.matmul(alpha.permute(0, 3, 1, 2), value_l)

        # x = [N, n_head, L, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [N, L, n_head, head_dim]

        x = x.view(N, -1, self.node_dim)

        # x = [N, L, node_dim]

        return x

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        node_from_pair = alpha.unsqueeze(-1) * z.unsqueeze(-2)

        # node_from_pair = [N, L, L, n_head, pair_dim]

        node_from_pair = node_from_pair.sum(dim=2).reshape(N, L, -1)

        # node_from_pair = [N, L, n_head * pair_dim]

        return node_from_pair

    def forward(self, x, z, plddt):

        # x = [N, L, node_dim]
        # z = [N, L, L, pair_dim]
        # plddt = [N, L, L, 1]

        alpha_from_node = self._node2alpha(x)
        alpha_from_pair = self._pair2alpha(z)

        # alpha_from_node = [N, L, L, n_head]
        # alpha_from_pair = [N, L, L, n_head]

        alpha = self.conv2dalpha(torch.cat((alpha_from_pair, alpha_from_node), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        alpha = self.alpha_mask_plddt(torch.cat((alpha, plddt), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # alpha = [N, L, L, n_head]

        node_from_node = self._node_aggregation(alpha, x)
        node_from_pair = self._pair_aggregation(alpha, z)

        # node_from_node = [N, L, node_dim]
        # node_from_pair = [N, L, n_head * pair_dim]

        x_out = self.out_transform(torch.cat([node_from_pair, node_from_node], dim=-1))
        x = self.layer_norm(x + x_out)
        return x, self.alpha2pair(torch.cat((z, alpha), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class PretrainEncoder(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim, num_layer):
        super().__init__()

        self.esm2_transform = torch.nn.Sequential(torch.nn.LayerNorm(1280), torch.nn.Linear(1280, 640), torch.nn.LeakyReLU(), torch.nn.Linear(640, 320), torch.nn.LeakyReLU(), torch.nn.Linear(320, node_dim), torch.nn.LeakyReLU(), torch.nn.Linear(node_dim, node_dim))
        self.pair_encoder = torch.nn.Linear(3, pair_dim)
        self.blocks = torch.nn.ModuleList([PretrainGAT(node_dim, n_head, pair_dim) for _ in range(num_layer)])
        self.pair_transform = torch.nn.Sequential(torch.nn.LayerNorm(pair_dim), torch.nn.Linear(pair_dim, pair_dim * 2), torch.nn.LeakyReLU(), torch.nn.Linear(pair_dim * 2, pair_dim), torch.nn.LeakyReLU())

    def forward(self, embedding, pair, plddt):

        # plddt = [N, L, L]
        # embedding = [N, L, 1280]
        # pair = [N, L, L, 3]

        embedding = self.esm2_transform(embedding)

        # embedding = [N, L, node_dim]

        pair = self.pair_encoder(pair)

        # pair = [N, L, L, pair_dim]

        for block in self.blocks:
            embedding, pair = block(embedding, pair, plddt.unsqueeze(-1))

        return embedding, self.pair_transform(pair)


class PretrainModel(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim, num_layer):
        super().__init__()

        self.pretrain_encoder = PretrainEncoder(node_dim, n_head, pair_dim, num_layer)
        self.pretrain_single_mlp = torch.nn.Linear(node_dim, 20)
        self.pretrain_double_mlp = torch.nn.Sequential(torch.nn.InstanceNorm2d(pair_dim), torch.nn.Conv2d(pair_dim, 200, 3, 1, 1), torch.nn.LeakyReLU(), torch.nn.InstanceNorm2d(200), torch.nn.Conv2d(200, 400, 3, 1, 1), torch.nn.LeakyReLU())
        self.logits_coef1 = torch.nn.Parameter(torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], requires_grad=True))
        self.logits_coef2 = torch.nn.Parameter(torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], requires_grad=True))

    def forward(self, data):

        single_feat, double_feat = self.pretrain_encoder(data["embedding"], data["pair"], data["plddt"])

        single_logits = self.pretrain_single_mlp(single_feat)
        single_logits = self.logits_coef1[None, :, None, None] * torch.cat((single_logits.unsqueeze(1), data["single_logits"]), dim=1)
        single_logits = single_logits.mean(1)

        double_logits = self.pretrain_double_mlp(double_feat.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        double_logits = self.logits_coef2[None, :, None, None, None] * torch.cat((double_logits.unsqueeze(1), data["double_logits"]), dim=1)
        double_logits = double_logits.mean(1)
        return single_logits, double_logits
