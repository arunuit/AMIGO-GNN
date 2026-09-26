from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

@dataclass
class AMRData:
    graphs: list
    labels: np.ndarray
    sample_ids: list
    feature_names: list
    label_encoder: LabelEncoder
    scaler: StandardScaler | None
    split_ids: dict

def _numeric_feature_names(nodes: pd.DataFrame):
    reserved = {"node_id", "node_type"}
    return [c for c in nodes.columns if c not in reserved and pd.api.types.is_numeric_dtype(nodes[c])]

def load_tables(cfg):
    samples = pd.read_csv(cfg["data"]["samples"])
    nodes = pd.read_csv(cfg["data"]["nodes"])
    edges = pd.read_csv(cfg["data"]["edges"])
    sample_nodes = pd.read_csv(cfg["data"]["sample_nodes"])
    required = {
        "samples": {"sample_id", "label"},
        "nodes": {"node_id", "node_type"},
        "edges": {"source", "target", "relation"},
        "sample_nodes": {"sample_id", "node_id"},
    }
    for name, cols in required.items():
        df = locals()[name]
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"{name}.csv missing columns: {sorted(missing)}")
    return samples, nodes, edges, sample_nodes

def stratified_split(samples, seed=42, train=0.70, val=0.15, test=0.15):
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("train + val + test must equal 1.")
    train_df, temp = train_test_split(
        samples, test_size=(1-train), random_state=seed,
        stratify=samples["label"]
    )
    relative_test = test / (val + test)
    val_df, test_df = train_test_split(
        temp, test_size=relative_test, random_state=seed,
        stratify=temp["label"]
    )
    return {
        "train": train_df["sample_id"].tolist(),
        "val": val_df["sample_id"].tolist(),
        "test": test_df["sample_id"].tolist(),
    }

def _build_edge_index(node_ids, edges):
    idx = {n:i for i,n in enumerate(node_ids)}
    pairs = []
    for _, r in edges.iterrows():
        s, t = r["source"], r["target"]
        if s in idx and t in idx:
            pairs.append((idx[s], idx[t]))
            pairs.append((idx[t], idx[s]))
    if not pairs:
        # self-loops make isolated-node samples valid
        pairs = [(i, i) for i in range(len(node_ids))]
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()

def build_graphs(cfg, split_seed=42):
    samples, nodes, edges, sample_nodes = load_tables(cfg)
    feature_names = _numeric_feature_names(nodes)
    if not feature_names:
        raise ValueError("nodes.csv must contain at least one numeric feature column.")

    split_ids = stratified_split(
        samples, seed=split_seed,
        train=cfg["split"]["train"], val=cfg["split"]["val"], test=cfg["split"]["test"]
    )

    # Fit scaling on TRAIN nodes only to reduce leakage.
    train_sample_nodes = sample_nodes[sample_nodes["sample_id"].isin(split_ids["train"])]
    train_node_ids = train_sample_nodes["node_id"].unique()
    train_nodes = nodes[nodes["node_id"].isin(train_node_ids)]
    scaler = StandardScaler()
    scaler.fit(train_nodes[feature_names].fillna(0.0).values)

    node_map = nodes.set_index("node_id")
    label_encoder = LabelEncoder()
    label_encoder.fit(samples["label"])

    graphs, labels, ids = [], [], []
    for _, row in samples.iterrows():
        sid = row["sample_id"]
        ids_for_sample = sample_nodes.loc[
            sample_nodes["sample_id"] == sid, "node_id"
        ].tolist()
        sub = nodes[nodes["node_id"].isin(ids_for_sample)].copy()
        if sub.empty:
            raise ValueError(f"No nodes found for sample {sid}.")
        x = scaler.transform(sub[feature_names].fillna(0.0).values)
        e = _build_edge_index(sub["node_id"].tolist(), edges)
        y = label_encoder.transform([row["label"]])[0]
        graphs.append(Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=e,
            y=torch.tensor([y], dtype=torch.long),
            sample_id=sid
        ))
        labels.append(y)
        ids.append(sid)

    return AMRData(graphs, np.asarray(labels), ids, feature_names, label_encoder, scaler, split_ids)

def get_split_indices(data: AMRData, split_name: str):
    wanted = set(data.split_ids[split_name])
    return [i for i, sid in enumerate(data.sample_ids) if sid in wanted]
