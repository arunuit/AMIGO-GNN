from __future__ import annotations
import copy
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score

def loaders(graphs, indices, batch_size=16, shuffle=False):
    return DataLoader([graphs[i] for i in indices], batch_size=batch_size, shuffle=shuffle)

def train_model(model, train_loader, val_loader, device, lr=1e-3,
                weight_decay=1e-5, epochs=200, patience=20):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_val, wait = None, float("inf"), 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(1, epochs+1):
        model.train()
        losses, yt, yp = [], [], []
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = torch.nn.functional.cross_entropy(logits, batch.y.view(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())
            yt.extend(batch.y.view(-1).detach().cpu().numpy())
            yp.extend(logits.argmax(1).detach().cpu().numpy())
        model.eval()
        val_losses, vyt, vyp = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                loss = torch.nn.functional.cross_entropy(logits, batch.y.view(-1))
                val_losses.append(loss.item())
                vyt.extend(batch.y.view(-1).cpu().numpy())
                vyp.extend(logits.argmax(1).cpu().numpy())
        tr_acc = accuracy_score(yt, yp)
        va_acc = accuracy_score(vyt, vyp)
        tr_loss, va_loss = np.mean(losses), np.mean(val_losses)
        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(va_loss))
        history["train_acc"].append(float(tr_acc))
        history["val_acc"].append(float(va_acc))
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history

def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            ys.extend(batch.y.view(-1).cpu().numpy())
            ps.append(torch.softmax(logits, dim=1).cpu().numpy())
    p = np.concatenate(ps)
    return np.asarray(ys), p

def metrics(y_true, proba):
    pred = proba.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="macro", zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }
    try:
        out["auc_ovr_macro"] = float(roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro"
        ))
    except ValueError:
        out["auc_ovr_macro"] = None
    return out, pred
