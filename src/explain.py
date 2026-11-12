from __future__ import annotations
import numpy as np
import torch
import shap
from torch_geometric.data import Batch

class GraphSHAP:
    """
    KernelSHAP wrapper for graph node-feature dimensions.
    The topology is kept fixed while feature dimensions are masked.
    """
    def __init__(self, model, graph, device, background_graphs=None):
        self.model = model
        self.graph = graph
        self.device = device
        self.base = graph.x.detach().cpu().numpy()
        self.background = background_graphs or [self.base]

    def _predict(self, flat):
        flat = np.asarray(flat)
        if flat.ndim == 1:
            flat = flat[None, :]
        outputs = []
        for row in flat:
            x = torch.tensor(row.reshape(self.base.shape), dtype=torch.float32)
            g = self.graph.clone()
            g.x = x
            b = Batch.from_data_list([g]).to(self.device)
            with torch.no_grad():
                p = torch.softmax(self.model(b.x, b.edge_index, b.batch), dim=1)
            outputs.append(p.cpu().numpy()[0])
        return np.asarray(outputs)

    def explain(self, nsamples=100, class_index=None):
        bg = np.asarray([x.reshape(-1) for x in self.background])
        explainer = shap.KernelExplainer(self._predict, bg)
        values = explainer.shap_values(self.base.reshape(1, -1), nsamples=nsamples)
        # shap can return list[class] or array depending on SHAP version.
        if isinstance(values, list):
            if class_index is None:
                class_index = int(self._predict(self.base.reshape(1,-1)).argmax(1)[0])
            vals = np.asarray(values[class_index])[0]
        else:
            vals = np.asarray(values)
            if vals.ndim == 3:
                if class_index is None:
                    class_index = int(self._predict(self.base.reshape(1,-1)).argmax(1)[0])
                vals = vals[0,:,class_index]
            else:
                vals = vals.reshape(-1)
        # Aggregate absolute attribution across nodes for each feature dimension.
        per_node = vals.reshape(self.base.shape)
        importance = np.mean(np.abs(per_node), axis=0)
        return importance
