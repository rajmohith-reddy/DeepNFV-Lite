"""Policy VNF: receives a traffic class from the classifier VNF and returns a network action.

NOTE: these rules are PROTOTYPE rules chosen for this project, not rules from the DeepNFV paper.
"""
import os

from flask import Flask, jsonify, request

app = Flask(__name__)

MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.60"))
RULES = {
    "Gmail": "ALLOW",
    "Outlook": "ALLOW",
    "Facetime": "PRIORITIZE",
    "BitTorrent": "RATE_LIMIT",
    "WorldOfWarcraft": "BLOCK"
}


@app.get("/health")
def health():
    return jsonify(status="ok", service="policy-vnf")


@app.post("/decide")
def decide():
    data = request.get_json(silent=True) or {}
    cls, conf = data.get("class"), data.get("confidence")
    if cls is None or conf is None:
        return jsonify(error="body must contain 'class' and 'confidence'"), 400
    if float(conf) < MIN_CONFIDENCE or cls not in RULES:
        action, reason = "BLOCK", f"unknown class or confidence < {MIN_CONFIDENCE}"
    else:
        action, reason = RULES[cls], f"rule for {cls}"
    return jsonify({"class": cls, "confidence": conf, "action": action, "reason": reason})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8001")))
