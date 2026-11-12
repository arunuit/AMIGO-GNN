import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AMIGO-GNN Dashboard", layout="wide")
st.title("AMIGO-GNN Interactive Decision Support Dashboard")
st.caption("Research visualization only — not a clinical diagnostic system.")

run_dir = Path(st.sidebar.text_input("Run directory", "runs/demo"))
if not run_dir.exists():
    st.warning("Run directory does not exist. Run the training script first.")
    st.stop()

metrics_file = run_dir/"metrics.json"
if metrics_file.exists():
    metrics=json.loads(metrics_file.read_text())
    cols=st.columns(len(metrics))
    for c,(k,v) in zip(cols,metrics.items()):
        c.metric(k.replace("_"," ").title(), f"{v:.4f}" if isinstance(v,float) else str(v))

for name,title in [("confusion_matrix.png","Confusion Matrix"),
                   ("roc.png","ROC Curves"),("pr.png","Precision–Recall Curves")]:
    p=run_dir/name
    if p.exists():
        st.subheader(title)
        st.image(str(p))

fp=run_dir/"shap_feature_importance.npy"
if fp.exists():
    st.subheader("SHAP Feature Importance")
    a=np.load(fp)
    imp=np.mean(np.abs(a),axis=0)
    meta=json.loads((run_dir/"metadata.json").read_text())
    names=meta["feature_names"]
    n=min(len(names),len(imp))
    df=pd.DataFrame({"feature":names[:n],"mean_abs_shap":imp[:n]}).sort_values("mean_abs_shap",ascending=False)
    st.dataframe(df.head(20),use_container_width=True)
