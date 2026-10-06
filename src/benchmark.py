"""Benchmark the running vNF chain: inference/chain latency, memory and (optionally) startup time.

Run against the Docker deployment for report numbers:
    docker compose up -d --build
    python src/benchmark.py --startup        # also measures `docker compose up` -> healthy
The JSON/plots record the measurement `mode` ("docker" or "local-process"); only "docker"
results describe the containerised deployment.
"""
import argparse
import json
import os
import re
import shutil
import statistics as st
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")


def summarise(v):
    v = sorted(v)
    return {"mean": st.mean(v), "median": st.median(v), "min": v[0], "max": v[-1],
            "p95": v[int(0.95 * (len(v) - 1))], "std": st.pstdev(v), "n": len(v)}


def time_requests(url, samples, n, chain):
    lat = []
    with requests.Session() as s:
        for i in range(n):
            body = {"packet_hex": samples[i % len(samples)]["packet_hex"]}
            t = time.perf_counter()
            r = s.post(f"{url}/predict?chain={'true' if chain else 'false'}", json=body, timeout=5)
            r.raise_for_status()
            lat.append((time.perf_counter() - t) * 1000)
    return lat


def docker_running():
    if not shutil.which("docker"):
        return False
    out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
    return "deepnfv-classifier" in out.stdout


def to_mib(s):
    m = re.match(r"([\d.]+)\s*([KMG]i?B)", s)
    v, u = float(m.group(1)), m.group(2)
    return v * {"KiB": 1 / 1024, "KB": 1 / 1024, "MiB": 1, "MB": 1, "GiB": 1024, "GB": 1024}[u]


def docker_memory():
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.MemUsage}}"],
                         capture_output=True, text=True).stdout
    mem = {}
    for line in out.strip().splitlines():
        name, usage = line.split("|")
        if name in ("deepnfv-classifier", "deepnfv-policy"):
            mem[name.replace("deepnfv-", "")] = to_mib(usage.split("/")[0].strip())
    return mem


def local_memory():
    import psutil
    mem = {}
    try:
        for p in psutil.process_iter(["cmdline", "memory_info"]):
            try:
                cmd = " ".join(p.info.get("cmdline") or [])
                for key in ("classifier", "policy"):
                    if f"{key}/app.py" in cmd or f"{key}\\app.py" in cmd:
                        mem[key] = p.info["memory_info"].rss / 2**20
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception:
        pass
    return mem


def measure_startup():
    subprocess.run(["docker", "compose", "down"], cwd=ROOT, capture_output=True)
    t = time.perf_counter()
    subprocess.run(["docker", "compose", "up", "-d", "--wait"], cwd=ROOT, check=True, capture_output=True)
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--startup", action="store_true", help="restart compose and time startup (docker mode only)")
    args = ap.parse_args()
    mode = "docker" if docker_running() else "local-process"
    res = {"mode": mode}
    if args.startup and mode == "docker":
        res["startup_s"] = round(measure_startup(), 2)
        time.sleep(3)
    samples = json.load(open(os.path.join(ROOT, "data", "processed", "samples.json")))
    time_requests(args.url, samples, 10, True)  # warm-up (not recorded)
    lat_only = time_requests(args.url, samples, args.n, False)
    lat_chain = time_requests(args.url, samples, args.n, True)
    res["classifier_only_ms"] = summarise(lat_only)
    res["full_chain_ms"] = summarise(lat_chain)
    res["memory_mib"] = docker_memory() if mode == "docker" else local_memory()
    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(os.path.join(RES, "benchmark.json"), "w"), indent=2)
    tag = "Docker containers" if mode == "docker" else "LOCAL PROCESSES (not Docker)"

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(lat_only, label="classifier only", alpha=.8)
    ax[0].plot(lat_chain, label="classifier -> policy chain", alpha=.8)
    ax[0].set(title="Per-request latency", xlabel="request #", ylabel="ms"); ax[0].legend(); ax[0].grid(alpha=.3)
    for k, (name, v) in enumerate([("classifier only", res["classifier_only_ms"]), ("full chain", res["full_chain_ms"])]):
        ax[1].bar(k, v["mean"], yerr=[[v["mean"] - v["min"]], [v["max"] - v["mean"]]], capsize=6,
                  color=["#2b6cb0", "#dd8452"][k])
        ax[1].text(k, v["max"] * 1.02, f"mean {v['mean']:.1f}\nmin {v['min']:.1f} / max {v['max']:.1f}", ha="center", fontsize=8)
    ax[1].set_xticks([0, 1], ["classifier only", "full chain"])
    ax[1].set(title="Mean latency (bars: min-max)", ylabel="ms"); ax[1].set_ylim(0, res["full_chain_ms"]["max"] * 1.35)
    fig.suptitle(f"Inference latency, n={args.n} - {tag}", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "latency.png"), dpi=200); plt.close(fig)

    if res["memory_mib"]:
        fig, ax = plt.subplots(figsize=(4.8, 4))
        names, vals = list(res["memory_mib"]), list(res["memory_mib"].values())
        bars = ax.bar(names, vals, color=["#2b6cb0", "#dd8452"][:len(vals)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom")
        ax.set(ylabel="MiB", title=f"Memory usage - {tag}")
        ax.title.set_fontsize(9); fig.tight_layout(); fig.savefig(os.path.join(RES, "memory.png"), dpi=200); plt.close(fig)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
