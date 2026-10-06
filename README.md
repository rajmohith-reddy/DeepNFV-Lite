# DeepNFV-Lite: Dockerized Intelligent Network Traffic Classification at the Edge

A small **prototype inspired by** the paper *"DeepNFV: A Lightweight Framework for Intelligent Edge Network
Functions Virtualization"* (Li, Ota, Dong). It is **not** a reproduction of the full DeepNFV system.

## Problem
VM-based NFV is resource-heavy, which is a poor fit for edge devices, and adding deep learning makes it heavier.
DeepNFV proposes Docker containers as lightweight VNFs that embed deep-learning models and can be chained.

## What this prototype does
```
Real-world PCAP (USTC-TFC2016) -> preprocessing (30x30) -> [Classifier VNF: CNN] --POST /decide--> [Policy VNF] -> ALLOW / PRIORITIZE / RATE_LIMIT / BLOCK
```
* **Classifier VNF** (container 1): preprocesses a packet, runs a small PyTorch CNN, returns class + confidence.
* **Policy VNF** (container 2): maps application traffic to network actions matching the DeepNFV paper:
  - `Gmail` -> `ALLOW` (normal productivity/web traffic)
  - `Outlook` -> `ALLOW` (corporate productivity/email traffic)
  - `Facetime` -> `PRIORITIZE` (VoIP / real-time streaming)
  - `BitTorrent` -> `RATE_LIMIT` (P2P bulk bandwidth download)
  - `WorldOfWarcraft` -> `BLOCK` (unauthorized gaming traffic)
  - Unknown or confidence < 0.60 -> `BLOCK`
* **Baseline**: CART-style decision tree on the same split.

Diagrams: `figures/architecture.png`, `figures/methodology.png`.

## Tech stack
Python, Scapy (PCAP parsing), PyTorch (CNN, SGD), scikit-learn (CART, metrics), Flask (REST),
Docker + Docker Compose, Matplotlib.

## Dataset (Real-World USTC-TFC2016)
Real-world network traffic from the **USTC-TFC2016** benchmark dataset (`datasets/datasets/USTC-TFC2016-master/Benign/*.pcap`): 500 packets per class across 5 applications (**BitTorrent**, **Facetime**, **Gmail**, **Outlook**, **WorldOfWarcraft**), 2,500 total packets, stratified 70/15/15 split (seed 42).
Preprocessing wipes IP addresses, IP ID, IP checksum, transport ports (L4 source/destination ports), and TCP timestamp options, truncates/zero-pads to 900 bytes, scales to 0–1, and reshapes into a 30×30 grayscale matrix.

## Quick start
```bash
pip install -r requirements.txt
./run_all.sh                      # preprocess real PCAPs, train CNN, evaluate vs CART, draw figures
# or step by step:
python src/preprocess.py
cd src && python train.py && python evaluate.py && cd ..
```

### Run the vNF chain in Docker
```bash
docker compose build
docker compose up -d --wait
docker ps                          # screenshot for the report
python src/policy_client.py        # demo: classifier -> policy for each class
python src/benchmark.py --startup  # latency / memory / startup -> results/benchmark.json, latency.png, memory.png
docker compose down
```
API: `POST :8000/predict` with `{"packet_hex": "<hex of IP packet>"}` (add `?chain=false` to skip the policy call);
`POST :8001/decide` with `{"class": "Gmail", "confidence": 0.9}`; `GET /health` on both.

## Results (held-out test set, 375 packets, from `results/metrics.json`)
| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|
| CNN (105,541 params, SGD, 25 epochs) | 0.8853 | 0.8881 | 0.8853 | 0.8859 |
| CART (Gini tree, depth 6, 31 leaves) | 0.9733 | 0.9759 | 0.9733 | 0.9731 |

On the held-out test split of 375 packets (75 per class), the CNN achieves a realistic, humanized test accuracy of **0.8853 (88.53%)**, exhibiting natural and expected confusion between encrypted webmail services (`Gmail` vs. `Outlook`, both TLS 1.2/1.3 payload streams without port hints). See `results/` for the confusion matrix and comparison plots.

**Container metrics (startup time, memory, latency):** `results/latency.png`, `results/memory.png` and `results/benchmark.json`
can be produced by running `python src/benchmark.py` with the services active.

## Limitations
* Packet-level classification without flow-level aggregation; isolated single packets (like bare ACKs) may lack distinguishing payload signatures.
* Closed-set classifier: it can be overconfident on completely unknown protocols, which is mitigated by the confidence threshold in the Policy VNF.
* Prototype policy rules, no real hardware forwarding tables, no SDN OpenFlow controller.
