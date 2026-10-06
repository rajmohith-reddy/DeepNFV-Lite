"""Evaluate CNN vs CART on the SAME held-out test split; write metrics, tables and plots."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report

from baseline import predict_cart, train_cart
from model import TrafficCNN
from preprocess import CLASSES

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, p), "precision": pr, "recall": rc, "f1": f1,
            "confusion_matrix": confusion_matrix(y, p, labels=range(len(CLASSES))).tolist(),
            "per_class": classification_report(y, p, target_names=CLASSES, output_dict=True, zero_division=0)}


def main():
    d = np.load(os.path.join(ROOT, "data", "processed", "dataset.npz"))
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]

    model = TrafficCNN(len(CLASSES))
    model.load_state_dict(torch.load(os.path.join(ROOT, "classifier", "model.pth"), map_location="cpu"))
    model.eval()
    with torch.no_grad():
        p_cnn = model(torch.from_numpy(Xte).unsqueeze(1).float()).argmax(1).numpy()
    cart = train_cart(Xtr, ytr)
    p_cart = predict_cart(cart, Xte)

    out = {"CNN": metrics(yte, p_cnn), "CART": metrics(yte, p_cart),
           "split": {"train": len(ytr), "val": len(d["y_val"]), "test": len(yte)},
           "class_counts": {"train": np.bincount(ytr).tolist(), "val": np.bincount(d["y_val"]).tolist(),
                            "test": np.bincount(yte).tolist(), "classes": CLASSES},
           "cart_depth": int(cart.get_depth()), "cart_leaves": int(cart.get_n_leaves())}
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    # --- markdown tables (paste into the report / README) ---
    lines = ["| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |", "|---|---:|---:|---:|---:|"]
    for m in ("CNN", "CART"):
        r = out[m]
        lines.append(f"| {m} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
    lines += ["", "| Class | Train | Val | Test |", "|---|---:|---:|---:|"]
    for i, c in enumerate(CLASSES):
        cc = out["class_counts"]
        lines.append(f"| {c} | {cc['train'][i]} | {cc['val'][i]} | {cc['test'][i]} |")
    with open(os.path.join(RES, "results_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # --- plots ---
    names = ["CNN", "CART"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    acc = [out[m]["accuracy"] for m in names]
    bars = ax.bar(names, acc, color=["#2b6cb0", "#dd8452"])
    ax.set(ylim=(0, 1.08), ylabel="accuracy", title="Test accuracy: CNN vs CART")
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "accuracy.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    w, x = 0.35, np.arange(3)
    for k, (m, col) in enumerate(zip(names, ["#2b6cb0", "#dd8452"])):
        vals = [out[m][s] for s in ("precision", "recall", "f1")]
        bars = ax.bar(x + (k - .5) * w, vals, w, label=m, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["Precision", "Recall", "F1-score"]); ax.set(ylim=(0, 1.1), title="Macro-averaged metrics (test set)")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(os.path.join(RES, "precision_recall_f1.png"), dpi=200); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, m in zip(axs, names):
        cm = np.array(out[m]["confusion_matrix"])
        a.imshow(cm, cmap="Blues")
        n_c = len(CLASSES)
        a.set(title=f"{m} confusion matrix", xticks=range(n_c), yticks=range(n_c), xticklabels=CLASSES,
              yticklabels=CLASSES, xlabel="Predicted", ylabel="Actual")
        plt.setp(a.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        plt.setp(a.get_yticklabels(), fontsize=8)
        for i in range(n_c):
            for j in range(n_c):
                a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "confusion_matrix.png"), dpi=200); plt.close(fig)
    print("saved plots to results/")


if __name__ == "__main__":
    main()
