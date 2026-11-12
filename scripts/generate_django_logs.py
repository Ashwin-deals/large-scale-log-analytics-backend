"""Generate SYNTHETIC Django request/error logs with injected faults.

There is no real Django log in this project, so this writes one: each user
(the block) makes requests over a few days -- mostly 2xx/3xx with
lognormal latency, a little background 404/500/slow noise -- and a
fixed share of users get an injected fault:

    500_burst   10-40 requests within a few minutes, mostly 500 + ERROR lines
    slow        15-50 requests with multi-second latency + "Slow request" WARNING lines
    db_errors   a run of ERROR lines from django.db.backends (connection failures)

Everything is driven by --seed, so the same command reproduces the same file.

Writes (both marked SYNTHETIC in their names):
    data/raw/django/django_SYNTHETIC.log
    data/raw/django/django_SYNTHETIC_labels.csv   user_block,label,fault  (block = "user-<id>")

Usage:
    python3 scripts/generate_django_logs.py
    python3 scripts/generate_django_logs.py --users 3000 --days 5 --fault-rate 0.03 --seed 7
"""

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

OUT_DIR = Path("data/raw/django")
LOG_PATH = OUT_DIR / "django_SYNTHETIC.log"
LABELS_PATH = OUT_DIR / "django_SYNTHETIC_labels.csv"

ROUTES = [  # (method, path template, weight, typical bytes)
    ("GET", "/", 18, 5200), ("GET", "/api/products/{id}", 22, 1800), ("GET", "/api/products", 10, 9400),
    ("GET", "/api/orders/{id}", 8, 1500), ("POST", "/api/cart", 9, 420), ("GET", "/api/cart", 9, 1100),
    ("POST", "/api/checkout", 4, 380), ("GET", "/api/reports/{id}", 3, 20400), ("POST", "/accounts/login", 5, 310),
    ("GET", "/static/app.js", 12, 88000),
]
FAULT_TYPES = ("500_burst", "slow", "db_errors")
START = datetime(2026, 3, 2, 0, 0, 0)


def line(t, level, logger, user, ip, method, path, status, size, ms, detail=""):
    stamp = f"{t:%Y-%m-%d %H:%M:%S},{t.microsecond // 1000:03d}"
    tail = f" {detail}" if detail else ""
    return f'[{stamp}] {level} {logger} user={user} ip={ip} "{method} {path} HTTP/1.1" {status} {size} {ms}ms{tail}'


def request(rng, t, user, ip):
    method, template, _, size = rng.choices(ROUTES, weights=[r[2] for r in ROUTES])[0]
    path = template.format(id=rng.randint(1, 5000))
    ms = max(3, int(rng.lognormvariate(4.2, 0.6)))  # median ~65 ms
    roll = rng.random()
    if roll < 0.02:
        return line(t, "WARNING", "django.request", user, ip, method, path, 404, 180, ms, f"Not Found: {path}")
    if roll < 0.022:  # background noise: the odd 500 for a normal user
        return line(t, "ERROR", "django.request", user, ip, method, path, 500, 89, ms * 4, f"Internal Server Error: {path}")
    if roll < 0.025:
        ms = rng.randint(2000, 3500)
        return line(t, "WARNING", "app.performance", user, ip, method, path, 200, size, ms, "Slow request")
    status = 302 if method == "POST" and path == "/accounts/login" else 200
    return line(t, "INFO", "django.request", user, ip, method, path, status, int(size * rng.uniform(0.7, 1.3)), ms)


def fault_lines(rng, kind, t, user, ip):
    out = []
    if kind == "500_burst":
        for _ in range(rng.randint(10, 40)):
            t += timedelta(seconds=rng.uniform(1, 12))
            method, path = rng.choice([("POST", "/api/checkout"), ("GET", f"/api/orders/{rng.randint(1, 5000)}"), ("POST", "/api/cart")])
            if rng.random() < 0.85:
                out.append((t, line(t, "ERROR", "django.request", user, ip, method, path, 500, 89, rng.randint(300, 2500), f"Internal Server Error: {path}")))
            else:
                out.append((t, request(rng, t, user, ip)))
    elif kind == "slow":
        for _ in range(rng.randint(15, 50)):
            t += timedelta(seconds=rng.uniform(5, 60))
            path = f"/api/reports/{rng.randint(1, 5000)}" if rng.random() < 0.6 else f"/api/products/{rng.randint(1, 5000)}"
            out.append((t, line(t, "WARNING", "app.performance", user, ip, "GET", path, 200, rng.randint(15000, 40000), rng.randint(4000, 15000), "Slow request")))
    else:  # db_errors
        for _ in range(rng.randint(8, 30)):
            t += timedelta(seconds=rng.uniform(1, 20))
            path = f"/api/orders/{rng.randint(1, 5000)}"
            out.append((t, line(t, "ERROR", "django.db.backends", user, ip, "GET", path, 500, 89, rng.randint(1000, 5000),
                                "OperationalError: could not connect to server: Connection refused")))
    return out


def generate(users: int, days: int, fault_rate: float, seed: int):
    rng = random.Random(seed)
    events, labels = [], []
    faulty = set(rng.sample(range(1, users + 1), max(1, round(users * fault_rate))))
    for user in range(1, users + 1):
        ip = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        # a few sessions of a handful of requests each, spread over the period
        for _ in range(rng.randint(1, 6)):
            t = START + timedelta(seconds=rng.uniform(0, days * 86400))
            for _ in range(rng.randint(2, 25)):
                t += timedelta(seconds=rng.expovariate(1 / 20))
                events.append((t, request(rng, t, user, ip)))
        fault = None
        if user in faulty:
            fault = rng.choice(FAULT_TYPES)
            t = START + timedelta(seconds=rng.uniform(0, days * 86400))
            events.extend(fault_lines(rng, fault, t, user, ip))
        labels.append((f"user-{user}", "Anomaly" if fault else "Normal", fault or ""))
    events.sort(key=lambda item: item[0])
    return [text for _, text in events], labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--users", type=int, default=2000)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--fault-rate", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lines, labels = generate(args.users, args.days, args.fault_rate, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with LABELS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block_id", "label", "fault"])
        writer.writerows(labels)
    anomalies = sum(1 for _, label, _ in labels if label == "Anomaly")
    print(f"SYNTHETIC Django logs: {len(lines):,} lines, {len(labels):,} users, {anomalies} with injected faults (seed {args.seed})")
    print(f"Wrote {LOG_PATH}\nWrote {LABELS_PATH}")


if __name__ == "__main__":
    main()
