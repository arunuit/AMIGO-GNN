import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
import yaml

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs/demo")
    args=ap.parse_args()
    run=Path(args.run_dir)
    y=np.load(run/"y_true.npy")
    p=np.load(run/"proba.npy")
    pred=p.argmax(1)
    classes=json.loads((run/"metadata.json").read_text())["classes"]

    cm=confusion_matrix(y,pred)
    fig,ax=plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax,cmap="Blues",colorbar=False)
    fig.tight_layout(); fig.savefig(run/"confusion_matrix.png",dpi=300); plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,5))
    for i,c in enumerate(classes):
        fpr,tpr,_=roc_curve((y==i).astype(int),p[:,i])
        ax.plot(fpr,tpr,label=f"{c} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0,1],[0,1],"--",label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(); fig.tight_layout(); fig.savefig(run/"roc.png",dpi=300); plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,5))
    for i,c in enumerate(classes):
        pr,rc,_=precision_recall_curve((y==i).astype(int),p[:,i])
        apv=average_precision_score((y==i).astype(int),p[:,i])
        ax.plot(rc,pr,label=f"{c} (AP={apv:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(); fig.tight_layout(); fig.savefig(run/"pr.png",dpi=300); plt.close(fig)
    print("Saved confusion_matrix.png, roc.png and pr.png in",run)

if __name__=="__main__":
    main()
