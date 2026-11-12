from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Batch

def model_class(model, graph, device):
    model.eval()
    b = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        p = torch.softmax(model(b.x, b.edge_index, b.batch), dim=1)
    return int(p.argmax(1)[0].cpu())

def perturb_graph(graph, top_feature_idx, fill_value=0.0):
    g = graph.clone()
    keep = np.zeros(g.x.shape[1], dtype=bool)
    keep[np.asarray(top_feature_idx, dtype=int)] = True
    x = g.x.clone()
    x[:, ~keep] = fill_value
    g.x = x
    return g

def perturbation_fidelity(model, graphs, attributions, top_k=10, device="cpu"):
    """
    Fidelity = fraction of samples for which the model prediction is preserved
    after retaining only the selected explanation features.
    """
    kept = 0
    total = min(len(graphs), len(attributions))
    for i in range(total):
        attr = np.asarray(attributions[i])
        top = np.argsort(np.abs(attr))[-top_k:]
        original = model_class(model, graphs[i], device)
        explained = model_class(model, perturb_graph(graphs[i], top), device)
        kept += int(original == explained)
    return 100.0 * kept / max(total, 1)
