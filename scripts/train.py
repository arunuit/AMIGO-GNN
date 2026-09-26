from pathlib import Path
import argparse, yaml, json
import numpy as np
import torch
import networkx as nx
from torch_geometric.loader import DataLoader

from src.utils import seed_everything, device, save_json
from src.data import build_graphs, get_split_indices
from src.model import AMIGOGNN
from src.train_utils import loaders, train_model, predict, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--run_dir", default="runs/demo")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed_everything(cfg["seed"])
    dev = device()
    data = build_graphs(cfg, split_seed=cfg["seed"])
    tr = get_split_indices(data,"train")
    va = get_split_indices(data,"val")
    te = get_split_indices(data,"test")

    # A compact graph summary is used by BioMOS. Feature dimensions are shared
    # across all node types in this implementation.
    G = nx.Graph()
    for g in data.graphs:
        for n in range(g.x.size(0)):
            G.add_node(str(n))
        for s,t in g.edge_index.t().tolist():
            G.add_edge(str(s), str(t))

    model = AMIGOGNN(
        len(data.feature_names),
        hidden_dim=cfg["model"]["hidden_dim"],
        heads=cfg["model"]["heads"],
        layers=cfg["model"]["layers"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        bio_relu_lambda=cfg["model"]["bio_relu_lambda"],
    )

    train_loader = loaders(data.graphs,tr,cfg["training"]["batch_size"],True)
    val_loader = loaders(data.graphs,va,cfg["training"]["batch_size"],False)
    test_loader = loaders(data.graphs,te,cfg["training"]["batch_size"],False)

    history = train_model(
        model, train_loader, val_loader, dev,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"]
    )
    y_true, proba = predict(model, test_loader, dev)
    m, pred = metrics(y_true, proba)

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run/"model.pt")
    np.save(run/"y_true.npy", y_true)
    np.save(run/"proba.npy", proba)
    save_json(m, run/"metrics.json")
    save_json(history, run/"history.json")
    save_json({
        "feature_names": data.feature_names,
        "classes": data.label_encoder.classes_.tolist(),
        "split_ids": data.split_ids,
        "device": str(dev),
        "config": cfg
    }, run/"metadata.json")
    print(json.dumps(m, indent=2))

if __name__ == "__main__":
    main()
