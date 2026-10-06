"""Draw the architecture and methodology diagrams used in the report."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
BLUE, ORANGE, GREEN, GREY = "#dbeafe", "#fde7c8", "#d9f2e0", "#eeeeee"


def box(ax, x, y, w, h, text, fc=BLUE, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.6, color="#333"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center", fontsize=8, style="italic")


def architecture():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.2, 1.6, 1.9, 1.3, "Real-world\nPCAP traffic\n(USTC-TFC2016)", GREY)
    ax.add_patch(FancyBboxPatch((2.7, 0.5), 7.9, 3.6, boxstyle="round,pad=0.02,rounding_size=0.2", fc="white", ec="#555", ls="--", lw=1.4))
    ax.text(6.65, 3.85, "Docker network (docker compose)", ha="center", fontsize=9, color="#555")
    ax.add_patch(FancyBboxPatch((3.0, 0.8), 3.7, 2.8, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#f4f8ff", ec="#2b6cb0", lw=1.5))
    ax.text(4.85, 3.35, "Container 1: Classifier VNF", ha="center", fontsize=9.5, fontweight="bold", color="#2b6cb0")
    box(ax, 3.2, 1.9, 3.3, 0.9, "Preprocessing\nmask headers, pad/truncate, normalise, 30x30", BLUE, 8)
    box(ax, 3.2, 1.0, 3.3, 0.7, "CNN (PyTorch)  ->  class + confidence", BLUE, 8.5)
    ax.add_patch(FancyBboxPatch((7.6, 0.8), 2.8, 2.8, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#fffaf2", ec="#c0701f", lw=1.5))
    ax.text(9.0, 3.35, "Container 2: Policy VNF", ha="center", fontsize=9.5, fontweight="bold", color="#c0701f")
    box(ax, 7.8, 0.9, 2.4, 2.1, "Prototype rules\nGmail/Outlook -> ALLOW\nFacetime -> PRIORITIZE\nBitTorrent -> RATE_LIMIT\nWoW -> BLOCK\nlow conf. -> BLOCK", ORANGE, 7.5)
    arrow(ax, 2.1, 2.25, 3.2, 2.25, "POST /predict")
    arrow(ax, 6.7, 2.2, 7.6, 2.2, "POST /decide")
    ax.text(5.5, 0.15, "REST (JSON) between containers: the vNF chain", ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "architecture.png"), dpi=200); plt.close(fig)


def methodology():
    steps = [("Real-world PCAP dataset (USTC-TFC2016)\nBitTorrent / Facetime / Gmail / Outlook / WoW", GREY),
             ("Packet preprocessing\nextract IP bytes, mask L4 ports/headers,\npad/truncate to 900, normalise 0-1", BLUE),
             ("30 x 30 packet matrix\n70% train / 15% val / 15% test (seed 42)", BLUE),
             ("Training: CNN (SGD)   |   Baseline: CART decision tree", BLUE),
             ("Evaluation: accuracy, precision, recall, F1, confusion matrix", GREEN),
             ("Docker Classifier VNF (CNN inference) -> REST", ORANGE),
             ("Docker Policy VNF -> ALLOW / PRIORITIZE / RATE_LIMIT / BLOCK", ORANGE),
             ("Container benchmark: latency, memory, startup time", GREEN)]
    fig, ax = plt.subplots(figsize=(6.2, 9))
    ax.set_xlim(0, 6.2); ax.set_ylim(0, 9); ax.axis("off")
    h, gap, top = 0.85, 0.2, 8.8
    for i, (t, c) in enumerate(steps):
        y = top - (i + 1) * h - i * gap
        box(ax, 0.3, y, 5.6, h, t, c, 8.5)
        if i < len(steps) - 1:
            arrow(ax, 3.1, y, 3.1, y - gap)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "methodology.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    architecture(); methodology(); print("saved figures/architecture.png and figures/methodology.png")
