from pathlib import Path
import argparse, yaml, json
import numpy as np
import torch
import networkx as nx
import pandas as pd
from torch_geometric.loader import DataLoader

from src.utils import seed_everything, device, save_json
from src.data import build_graphs, get_split_indices
from src.model import AMIGOGNN
from src.biomos import BioMOS
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

    # BioMOS is fitted using TRAINING isolates only. In this generic CSV
    # implementation, its feature graph is a correlation graph over numeric
    # feature columns; it is not a substitute for a biologically curated
    # feature-to-ontology graph unless the input schema explicitly maps them.
    biomos_cfg = cfg.get("biomos", {})
    selected_feature_names = list(data.feature_names)
    biomos_history = []
    biomos_mask = np.ones(len(data.feature_names), dtype=bool)
    if biomos_cfg.get("enabled", False):
        train_graphs = [data.graphs[i] for i in tr]
        X_train = np.vstack([
            g.x.detach().cpu().numpy().mean(axis=0) for g in train_graphs
        ])
        y_train = np.asarray([int(g.y.item()) for g in train_graphs])
        # Build an explicit feature-correlation graph, avoiding the previous
        # mismatch between local node indices and feature-name identifiers.
        corr = np.atleast_2d(np.nan_to_num(np.corrcoef(X_train, rowvar=False), nan=0.0))
        feature_graph = nx.Graph()
        feature_graph.add_nodes_from(data.feature_names)
        threshold = float(biomos_cfg.get("correlation_graph_threshold", 0.10))
        for i in range(len(data.feature_names)):
            for j in range(i + 1, len(data.feature_names)):
                if abs(corr[i, j]) >= threshold:
                    feature_graph.add_edge(data.feature_names[i], data.feature_names[j],
                                           weight=float(abs(corr[i, j])))
        optimizer = BioMOS(
            population_size=biomos_cfg.get("population_size", 50),
            iterations=biomos_cfg.get("iterations", 100),
            alpha=biomos_cfg.get("alpha", 0.40),
            beta=biomos_cfg.get("beta", 0.25),
            gamma=biomos_cfg.get("gamma", 0.20),
            delta=biomos_cfg.get("delta", 0.15),
            graph_awareness=biomos_cfg.get("graph_awareness", 0.30),
            eval_epochs=biomos_cfg.get("eval_epochs", 5),
            seed=cfg["seed"],
        ).fit(X_train, y_train, feature_graph, data.feature_names)
        biomos_mask = optimizer.mask_
        selected_feature_names = [n for n, keep in zip(data.feature_names, biomos_mask) if keep]
        if not selected_feature_names:
            raise RuntimeError("BioMOS selected zero features; check data and optimizer settings.")
        for graph in data.graphs:
            graph.x = graph.x[:, biomos_mask].contiguous()
        biomos_history = optimizer.history


    model = AMIGOGNN(
        len(selected_feature_names),
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
    np.save(run/"biomos_feature_mask.npy", biomos_mask)
    if biomos_history:
        pd.DataFrame({"iteration": np.arange(1, len(biomos_history)+1),
                      "best_fitness": biomos_history}).to_csv(run/"biomos_history.csv", index=False)
    pd.DataFrame([{"split": split, "sample_id": sid}
                  for split, ids in data.split_ids.items() for sid in ids]).to_csv(
                      run/"split_ids.csv", index=False)
    save_json({
        "feature_names": data.feature_names,
        "selected_feature_names": selected_feature_names,
        "biomos_enabled": bool(biomos_cfg.get("enabled", False)),
        "classes": data.label_encoder.classes_.tolist(),
        "split_ids": data.split_ids,
        "device": str(dev),
        "config": cfg
    }, run/"metadata.json")
    print(json.dumps(m, indent=2))

if __name__ == "__main__":
    main()
