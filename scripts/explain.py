import argparse, json
from pathlib import Path
import numpy as np, torch
from torch_geometric.loader import DataLoader
import yaml
from src.utils import device
from src.data import build_graphs, get_split_indices
from src.model import AMIGOGNN
from src.explain import GraphSHAP

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run_dir",default="runs/demo")
    ap.add_argument("--samples",type=int,default=3)
    ap.add_argument("--nsamples",type=int,default=100)
    args=ap.parse_args()
    run=Path(args.run_dir)
    meta=json.loads((run/"metadata.json").read_text())
    cfg=meta["config"]
    data=build_graphs(cfg,split_seed=cfg["seed"])
    te=get_split_indices(data,"test")
    dev=device()
    model=AMIGOGNN(len(data.feature_names),
        hidden_dim=cfg["model"]["hidden_dim"],heads=cfg["model"]["heads"],
        layers=cfg["model"]["layers"],num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],bio_relu_lambda=cfg["model"]["bio_relu_lambda"]).to(dev)
    model.load_state_dict(torch.load(run/"model.pt",map_location=dev))
    model.eval()

    out=[]
    for idx in te[:args.samples]:
        bg=[data.graphs[j].x.numpy() for j in te[:min(5,len(te))]]
        expl=GraphSHAP(model,data.graphs[idx],dev,background_graphs=bg)
        imp=expl.explain(nsamples=args.nsamples)
        out.append(imp)
    np.save(run/"shap_feature_importance.npy",np.stack(out))
    print("Saved",run/"shap_feature_importance.npy")

if __name__=="__main__":
    main()
