"""Train the CNN with SGD on the preprocessed dataset and save classifier/model.pth."""
import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from model import TrafficCNN
from preprocess import CLASSES

ROOT = os.path.join(os.path.dirname(__file__), "..")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(X):
    return torch.from_numpy(X).unsqueeze(1).float()


def evaluate(model, X, y, loss_fn):
    model.eval()
    with torch.no_grad():
        out = model(X)
        return loss_fn(out, y).item(), (out.argmax(1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    torch.set_num_threads(1)

    d = np.load(os.path.join(ROOT, "data", "processed", "dataset.npz"))
    Xtr, ytr = to_tensor(d["X_train"]), torch.from_numpy(d["y_train"])
    Xva, yva = to_tensor(d["X_val"]), torch.from_numpy(d["y_val"])

    model = TrafficCNN(len(CLASSES))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TrafficCNN parameters: {n_params:,}")
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)  # SGD with momentum

    log = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_vl, best_val, best_state = float('inf'), 0.0, None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        tl, ta = evaluate(model, Xtr, ytr, loss_fn)
        vl, va = evaluate(model, Xva, yva, loss_fn)
        for k, v in zip(log, (ep, tl, ta, vl, va)):
            log[k].append(v)
        if vl < best_vl:  # checkpoint selection on validation LOSS (val accuracy saturates at 1.0 and ties)
            best_vl, best_val, best_state = vl, va, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:02d}  train_loss={tl:.4f} train_acc={ta:.4f}  val_loss={vl:.4f} val_acc={va:.4f}")

    log.update(best_val_loss=best_vl, train_time_s=round(time.time() - t0, 2), n_params=n_params, best_val_acc=best_val,
               epochs=args.epochs, lr=args.lr, batch=args.batch, seed=args.seed)
    torch.save(best_state, os.path.join(ROOT, "classifier", "model.pth"))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(log["epoch"], log["train_loss"], marker="o", label="train")
    ax[0].plot(log["epoch"], log["val_loss"], marker="s", label="validation")
    ax[0].set(title="CNN loss (SGD)", xlabel="epoch", ylabel="cross-entropy loss"); ax[0].legend()
    ax[1].plot(log["epoch"], log["train_acc"], marker="o", label="train")
    ax[1].plot(log["epoch"], log["val_acc"], marker="s", label="validation")
    ax[1].set(title="CNN accuracy", xlabel="epoch", ylabel="accuracy"); ax[1].legend()
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "training_curve.png"), dpi=200)
    print(f"saved classifier/model.pth (best val acc {best_val:.4f}, {log['train_time_s']}s)")


if __name__ == "__main__":
    main()
