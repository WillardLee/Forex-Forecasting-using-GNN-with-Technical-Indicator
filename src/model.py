import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATForecast(nn.Module):
    def __init__(self, node_feat_len, gat_hidden=64, gat_heads=4, mlp_hidden=64):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_len, gat_hidden)
        # GAT layers
        self.gat1 = GATConv(gat_hidden, gat_hidden // 2, heads=gat_heads, concat=True)
        self.gat2 = GATConv((gat_hidden // 2) * gat_heads, gat_hidden, heads=1, concat=False)
        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(gat_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU()
        )
        # outputs
        self.class_head = nn.Linear(mlp_hidden, 1)  # logits for BCEWithLogits
        self.ret_head = nn.Linear(mlp_hidden, 1)
        self.vol_head = nn.Linear(mlp_hidden, 1)

    def forward(self, x, edge_index, batch):
        # x: (total_nodes, node_feat_len)
        h = F.relu(self.node_proj(x))
        h = F.elu(self.gat1(h, edge_index))
        h = F.elu(self.gat2(h, edge_index))
        hg = self.pool(h, batch)
        z = self.mlp(hg)
        class_logit = self.class_head(z).squeeze(-1)
        ret = self.ret_head(z).squeeze(-1)
        vol = F.softplus(self.vol_head(z).squeeze(-1))  # ensure positive
        return class_logit, ret, vol

class LSTMBaseline(nn.Module):
    def __init__(self, num_nodes, window, hidden=64, mlp_hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_nodes, hidden_size=hidden, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden, mlp_hidden), nn.ReLU())
        self.class_head = nn.Linear(mlp_hidden, 1)
        self.ret_head = nn.Linear(mlp_hidden, 1)
        self.vol_head = nn.Linear(mlp_hidden, 1)

    def forward(self, node_windows_batch):
        # node_windows_batch: (B, window, num_nodes)
        out, _ = self.lstm(node_windows_batch)
        last = out[:, -1, :]
        z = self.mlp(last)
        return self.class_head(z).squeeze(-1), self.ret_head(z).squeeze(-1), F.softplus(self.vol_head(z).squeeze(-1))
