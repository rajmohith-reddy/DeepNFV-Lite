"""Classifier VNF: preprocess a packet, run the CNN, and forward the result to the policy VNF."""
import os
import sys
import time

import requests
import torch
from flask import Flask, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))  # local runs; in Docker the modules sit beside app.py
from model import TrafficCNN  # noqa: E402
from preprocess import CLASSES, bytes_to_matrix  # noqa: E402

torch.set_num_threads(1)
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(HERE, "model.pth"))
POLICY_URL = os.environ.get("POLICY_URL", "http://127.0.0.1:8001/decide")
policy_session = requests.Session()

model = TrafficCNN(len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", service="classifier-vnf", classes=CLASSES)


@app.post("/predict")
def predict():
    t0 = time.perf_counter()
    data = request.get_json(silent=True) or {}
    try:
        raw = bytes.fromhex(data["packet_hex"])
    except (KeyError, ValueError, TypeError):
        return jsonify(error="body must contain a valid hex string 'packet_hex'"), 400
    x = torch.from_numpy(bytes_to_matrix(raw)).view(1, 1, 30, 30)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    k = int(probs.argmax())
    result = {"class": CLASSES[k], "confidence": round(float(probs[k]), 4),
              "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)}}
    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    # forward to the policy VNF (the vNF chain) unless ?chain=false
    if request.args.get("chain", "true").lower() != "false":
        try:
            r = policy_session.post(POLICY_URL, json={"class": result["class"], "confidence": result["confidence"]}, timeout=2)
            result["policy"] = r.json()
        except requests.RequestException as e:
            result["policy"] = {"error": f"policy VNF unreachable: {e.__class__.__name__}"}
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
