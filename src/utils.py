from __future__ import annotations
import json, random, os
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
