"""Generate a small, labelled, SIMULATED PCAP dataset (DNS / ICMP / HTTP) with Scapy.

Labels come from the file the packets are written to (data/raw/<class>.pcap).
Header fields and payloads are randomised so packets within a class vary.
Everything is seeded for reproducibility.
"""
import argparse
import os
import random
import struct

from scapy.all import DNS, DNSQR, DNSRR, ICMP, IP, TCP, UDP, Raw, conf, wrpcap

conf.verb = 0

WORDS = ["mail", "cdn", "api", "shop", "news", "login", "static", "img", "video", "docs",
         "cloud", "edge", "iot", "sensor", "campus", "portal", "media", "update", "sync", "store"]
TLDS = ["com", "org", "net", "io", "edu", "in", "co.uk"]
PATHS = ["/", "/index.html", "/api/v1/items", "/login", "/search?q={w}", "/static/app.js",
         "/img/logo.png", "/products/{n}", "/users/{n}/profile", "/health", "/data.json"]
AGENTS = ["Mozilla/5.0 (X11; Linux x86_64)", "curl/8.4.0", "python-requests/2.31",
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "okhttp/4.12.0"]
CTYPES = ["text/html", "application/json", "image/png", "application/javascript", "text/plain"]


def rand_ip(rng):
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def rand_domain(rng):
    parts = [rng.choice(WORDS) + (str(rng.randint(1, 99)) if rng.random() < 0.4 else "")
             for _ in range(rng.randint(1, 3))]
    return ".".join(parts) + "." + rng.choice(TLDS)


def ip_layer(rng):
    return IP(src=rand_ip(rng), dst=rand_ip(rng), ttl=rng.choice([32, 56, 64, 117, 128, 255]),
              id=rng.randint(0, 65535), flags=rng.choice([0, 2]), tos=rng.choice([0, 0, 0, 16, 40]))


def eph_port(rng):
    return rng.randint(1024, 65535)


def make_dns(rng):
    domain = rand_domain(rng)
    if rng.random() < 0.4:  # response
        dns = DNS(id=rng.randint(0, 65535), qr=1, aa=rng.choice([0, 1]), rd=1, ra=1,
                  qd=DNSQR(qname=domain, qtype=1),
                  an=DNSRR(rrname=domain, type=1, ttl=rng.randint(30, 86400), rdata=rand_ip(rng)))
        sport, dport = 53, eph_port(rng)
    else:  # query
        dns = DNS(id=rng.randint(0, 65535), rd=1,
                  qd=DNSQR(qname=domain, qtype=rng.choice([1, 1, 1, 28, 15, 16])))
        sport, dport = eph_port(rng), 53
    if rng.random() < 0.10:  # DNS over TCP (2-byte length prefix)
        raw = bytes(dns)
        return (ip_layer(rng) / TCP(sport=sport, dport=dport, flags="PA", seq=rng.getrandbits(32),
                                    ack=rng.getrandbits(32), window=rng.choice([8192, 29200, 65535]))
                / Raw(struct.pack("!H", len(raw)) + raw))
    return ip_layer(rng) / UDP(sport=sport, dport=dport) / dns


def make_icmp(rng):
    if rng.random() < 0.9:  # echo request / reply
        n = rng.randint(8, 120)
        if rng.random() < 0.5:
            off = rng.randint(0, 255)
            payload = bytes((i + off) % 256 for i in range(n))
        else:
            payload = bytes(rng.getrandbits(8) for _ in range(n))
        icmp = ICMP(type=rng.choice([8, 0]), code=0, id=rng.randint(0, 65535), seq=rng.randint(0, 65535))
        return ip_layer(rng) / icmp / Raw(payload)
    return (ip_layer(rng) / ICMP(type=3, code=rng.choice([0, 1, 3, 13]))
            / Raw(bytes(rng.getrandbits(8) for _ in range(28))))


def make_http(rng):
    host = rand_domain(rng)
    server_port = rng.choice([80, 80, 8080, 8000])
    from_client = rng.random() < 0.55
    sport, dport = (eph_port(rng), server_port) if from_client else (server_port, eph_port(rng))
    if rng.random() < 0.15:  # bare TCP control segment, no payload
        flags, payload = rng.choice(["S", "A", "FA", "SA"]), b""
    elif from_client:
        path = rng.choice(PATHS).format(w=rng.choice(WORDS), n=rng.randint(1, 9999))
        if rng.random() < 0.25:
            body = ("data=" + "".join(rng.choice("abcdef0123456789") for _ in range(rng.randint(5, 60)))).encode()
            payload = (f"POST {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: {rng.choice(AGENTS)}\r\n"
                       f"Content-Type: application/x-www-form-urlencoded\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
        else:
            payload = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: {rng.choice(AGENTS)}\r\n"
                       f"Accept: */*\r\nConnection: {rng.choice(['keep-alive', 'close'])}\r\n\r\n").encode()
        flags = "PA"
    else:
        body = bytes(rng.choice(b"abcdefghij <>/=\"") for _ in range(rng.randint(0, 200)))
        payload = (f"HTTP/1.1 {rng.choice([200, 200, 200, 301, 404, 500])} OK\r\nServer: nginx\r\n"
                   f"Content-Type: {rng.choice(CTYPES)}\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
        flags = "PA"
    pkt = ip_layer(rng) / TCP(sport=sport, dport=dport, flags=flags, seq=rng.getrandbits(32),
                              ack=rng.getrandbits(32), window=rng.choice([8192, 29200, 64240, 65535]))
    return pkt / Raw(payload) if payload else pkt


GENERATORS = {"dns": make_dns, "icmp": make_icmp, "http": make_http}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    for name, gen in GENERATORS.items():
        pkts = [gen(rng) for _ in range(args.per_class)]
        path = os.path.join(args.out, f"{name}.pcap")
        wrpcap(path, pkts)
        print(f"wrote {len(pkts)} packets -> {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
