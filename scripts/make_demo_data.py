from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n = 180
classes = ["Resistant", "Intermediate", "Susceptible"]
samples = []
nodes = []
edges = []
sample_nodes = []

feature_defs = [
    ("gene_blaCTX_M","Resistance Gene"), ("gene_blaKPC","Resistance Gene"),
    ("gene_mecA","Resistance Gene"), ("gene_blaOXA_23","Resistance Gene"),
    ("drug_cefotaxime","Antibiotic"), ("drug_meropenem","Antibiotic"),
    ("mech_carbapenemase","Resistance Mechanism"),
    ("mech_ESBL","Resistance Mechanism"),
    ("mech_efflux","Resistance Mechanism"),
    ("path_beta_lactam","Biological Pathway"),
    ("path_carbapenem","Biological Pathway"),
    ("path_cell_wall","Biological Pathway"),
]
# 4 numeric attributes per biological feature node.
feature_names = [f"f{i}" for i in range(4)]

node_id = 0
for i in range(n):
    label = classes[i % 3]
    sid = f"iso_{i:04d}"
    species = ["Escherichia coli","Klebsiella pneumoniae","Staphylococcus aureus"][i % 3]
    samples.append((sid,label,species))
    ids = []
    for j,(name,ntype) in enumerate(feature_defs):
        nid = f"n_{node_id}"
        node_id += 1
        # Signal structure is only for a smoke test, not a biological dataset.
        base = np.zeros(4)
        if label == "Resistant":
            base[0] = 1.5 if j < 4 else 0.7
        elif label == "Intermediate":
            base[1] = 1.2 if j in [0,4,7] else 0.5
        else:
            base[2] = 1.3 if j in [4,5,9] else 0.3
        vals = base + rng.normal(0,0.35,4)
        nodes.append((nid,ntype,*vals))
        sample_nodes.append((sid,nid))
        ids.append(nid)
    # chain + skip edges within the sample graph
    for a,b in zip(ids[:-1],ids[1:]):
        relation = "associated_with"
        edges.append((a,b,relation))
    for a,b in zip(ids[:-2],ids[2:]):
        edges.append((a,b,"participates_in"))

pd.DataFrame(samples,columns=["sample_id","label","species"]).to_csv(RAW/"samples.csv",index=False)
pd.DataFrame(nodes,columns=["node_id","node_type"]+feature_names).to_csv(RAW/"nodes.csv",index=False)
pd.DataFrame(edges,columns=["source","target","relation"]).drop_duplicates().to_csv(RAW/"edges.csv",index=False)
pd.DataFrame(sample_nodes,columns=["sample_id","node_id"]).to_csv(RAW/"sample_nodes.csv",index=False)
print("Demo data written to", RAW)
