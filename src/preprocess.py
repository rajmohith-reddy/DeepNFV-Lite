"""Packet preprocessing: raw packet bytes -> 30x30 normalised matrix (DeepNFV-style CNN input).

Pipeline per packet:
  1. take the IP packet bytes
  2. header modification: zero IP src/dst, IP ID, IP checksum and the transport checksum,
     so the model cannot memorise per-host / random header fields
  3. truncate or zero-pad to 900 values
  4. normalise 0-255 -> 0-1
  5. reshape to 30 x 30

`bytes_to_matrix` has no scapy dependency so it also runs inside the classifier container.
"""
import argparse
import json
import os

import numpy as np

CLASSES = ["BitTorrent", "Facetime", "Gmail", "Outlook", "WorldOfWarcraft"]
SIDE = 30
N_VALUES = SIDE * SIDE  # 900
PACKETS_PER_CLASS = 500


def mask_headers(raw: bytes) -> bytearray:
    b = bytearray(raw)
    if len(b) < 20 or (b[0] >> 4) != 4:
        return b
    ihl = (b[0] & 0x0F) * 4
    b[4:6] = b"\x00\x00"       # identification
    b[10:12] = b"\x00\x00"     # header checksum
    b[12:20] = bytes(8)        # source + destination address
    proto = b[9]
    if proto == 6 and len(b) >= ihl + 20:  # TCP
        tcp_len = ((b[ihl + 12] >> 4) & 0x0F) * 4
        b[ihl : ihl + 4] = bytes(4)        # transport ports
        # Mask TCP timestamp option if present (kind 8)
        opt_idx = ihl + 20
        while opt_idx < ihl + tcp_len and opt_idx < len(b):
            opt_kind = b[opt_idx]
            if opt_kind in (0, 1):
                opt_idx += 1
            elif opt_kind == 8:
                opt_len = b[opt_idx + 1] if opt_idx + 1 < len(b) else 10
                b[opt_idx + 2 : min(opt_idx + opt_len, len(b))] = bytes(min(opt_len - 2, len(b) - opt_idx - 2))
                opt_idx += opt_len
            else:
                opt_len = b[opt_idx + 1] if opt_idx + 1 < len(b) else 1
                opt_idx += max(opt_len, 1)
    elif proto == 17 and len(b) >= ihl + 8:  # UDP
        b[ihl : ihl + 4] = bytes(4)        # transport ports
    off = {6: ihl + 16, 17: ihl + 6, 1: ihl + 2}.get(proto)  # TCP / UDP / ICMP checksum
    if off is not None and off + 2 <= len(b):
        b[off:off + 2] = b"\x00\x00"
    return b


def bytes_to_matrix(raw: bytes) -> np.ndarray:
    b = mask_headers(raw)[:N_VALUES]
    arr = np.zeros(N_VALUES, dtype=np.float32)
    arr[:len(b)] = np.frombuffer(bytes(b), dtype=np.uint8)
    return (arr / 255.0).reshape(SIDE, SIDE)


def find_dataset_dir(root):
    candidates = [
        os.path.join(root, "datasets", "datasets", "USTC-TFC2016-master"),
        os.path.join(root, "datasets", "USTC-TFC2016-master"),
        os.path.join(root, "dataset", "datasets", "USTC-TFC2016-master"),
        os.path.join(root, "dataset", "USTC-TFC2016-master"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not find USTC-TFC2016 dataset directory in {candidates}")


def read_pcap(path, max_packets=PACKETS_PER_CLASS, stride=2):
    from scapy.all import IP, PcapReader
    pkts = []
    with PcapReader(path) as reader:
        for i, p in enumerate(reader):
            if p.haslayer(IP):
                if i % stride == 0:
                    pkts.append(bytes(p[IP]))
                    if len(pkts) >= max_packets:
                        break
    return pkts


def main():
    from sklearn.model_selection import train_test_split
    root = os.path.join(os.path.dirname(__file__), "..")
    dataset_dir = find_dataset_dir(root)

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=dataset_dir)
    ap.add_argument("--out", default=os.path.join(root, "data", "processed"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-class", type=int, default=PACKETS_PER_CLASS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raws, labels = [], []
    for idx, name in enumerate(CLASSES):
        pcap_path = os.path.join(args.raw, "Benign", f"{name}.pcap")
        if not os.path.exists(pcap_path):
            pcap_path = os.path.join(args.raw, "Malware", f"{name}.pcap")
        pkts = read_pcap(pcap_path, max_packets=args.max_per_class)
        raws += pkts
        labels += [idx] * len(pkts)
        print(f"{name}: {len(pkts)} packets from {os.path.basename(pcap_path)}")

    X = np.stack([bytes_to_matrix(r) for r in raws])
    y = np.array(labels, dtype=np.int64)
    idx = np.arange(len(y))

    # 70 / 15 / 15 stratified split with a fixed seed
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=args.seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=args.seed)
    np.savez_compressed(os.path.join(args.out, "dataset.npz"),
                        X_train=X[tr], y_train=y[tr], X_val=X[va], y_val=y[va], X_test=X[te], y_test=y[te])

    # test packets as hex, used for the live demo and for benchmarking the containers
    samples = [{"label": CLASSES[y[i]], "packet_hex": raws[i].hex()} for i in te]
    with open(os.path.join(args.out, "samples.json"), "w") as f:
        json.dump(samples, f)
    print(f"train={len(tr)} val={len(va)} test={len(te)}  X shape={X.shape}")


if __name__ == "__main__":
    main()
