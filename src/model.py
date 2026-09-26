from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class BioReLU(nn.Module):
    """
    Manuscript definition:
        y=x, x>=0
        y=lambda*x*exp(x), x<0
    A clamp is used on x before exp() for numerical stability.
    """
    def __init__(self, lam=0.10):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        neg = self.lam * x * torch.exp(torch.clamp(x, min=-20.0, max=5.0))
        return torch.where(x >= 0, x, neg)

class AMIGOGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, heads=8, layers=2,
                 num_classes=3, dropout=0.5, bio_relu_lambda=0.10):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(max(0, layers-1)):
            self.convs.append(
                GATConv(hidden_dim*heads, hidden_dim, heads=heads, dropout=dropout)
            )
        out_dim = hidden_dim * heads
        self.bio_relu = BioReLU(bio_relu_lambda)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.bio_relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch)
        return self.classifier(g)

    @torch.no_grad()
    def predict_proba(self, data_loader, device):
        self.eval()
        out = []
        for batch in data_loader:
            batch = batch.to(device)
            logits = self(batch.x, batch.edge_index, batch.batch)
            out.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(out, dim=0).numpy()
