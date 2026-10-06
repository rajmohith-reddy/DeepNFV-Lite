"""Demo: send one sample packet of each class to the classifier VNF and print the full vNF-chain result."""
import argparse
import json
import os

import requests

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()
    samples = json.load(open(os.path.join(ROOT, "data", "processed", "samples.json")))
    seen = set()
    for s in samples:
        if s["label"] in seen:
            continue
        seen.add(s["label"])
        r = requests.post(f"{args.url}/predict", json={"packet_hex": s["packet_hex"]}, timeout=5).json()
        print(f"Input packet (true class): {s['label']}")
        print(f"  Classifier VNF -> predicted={r['class']}  confidence={r['confidence']}")
        print(f"  Policy VNF     -> action={r['policy'].get('action')}  ({r['policy'].get('reason')})\n")
    # exercise the BLOCK rule directly on the policy VNF (low confidence / unknown class)
    pol = args.url.replace(":8000", ":8001")
    r = requests.post(f"{pol}/decide", json={"class": "UNKNOWN", "confidence": 0.30}, timeout=5).json()
    print("Direct policy test: class=UNKNOWN confidence=0.30")
    print(f"  Policy VNF     -> action={r['action']}  ({r['reason']})")


if __name__ == "__main__":
    main()
