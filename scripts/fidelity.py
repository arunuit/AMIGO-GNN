import argparse, json
from pathlib import Path
import numpy as np, torch
from src.utils import device
from src.data import build_graphs, get_split_indices
from src.model import AMIGOGNN
from src.fidelity import perturbation_fidelity

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run_dir",default="runs/demo")
    ap.add_argument("--samples",type=int,default=50)
    ap.add_argument("--top_k",type=int,default=10)
    args=ap.parse_args()
    run=Path(args.run_dir)
    meta=json.loads((run/"metadata.json").read_text())
    cfg=meta["config"]
    data=build_graphs(cfg,split_seed=cfg["seed"])
    te=get_split_indices(data,"test")
    attr=np.load(run/"shap_feature_importance.npy")
    dev=device()
    model=AMIGOGNN(len(data.feature_names),
        hidden_dim=cfg["model"]["hidden_dim"],heads=cfg["model"]["heads"],
        layers=cfg["model"]["layers"],num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],bio_relu_lambda=cfg["model"]["bio_relu_lambda"]).to(dev)
    model.load_state_dict(torch.load(run/"model.pt",map_location=dev))
    value=perturbation_fidelity(model,[data.graphs[i] for i in te[:len(attr)]],attr,args.top_k,dev)
    print(f"Perturbation fidelity: {value:.2f}%")
    (run/"fidelity.json").write_text(json.dumps({"fidelity_percent":value,"top_k":args.top_k},indent=2))

if __name__=="__main__":
    main()
