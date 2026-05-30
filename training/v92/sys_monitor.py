"""Lightweight system + GPU monitor for v92 training.

Uses psutil for CPU/RAM and subprocess(nvidia-smi) for GPU stats — no extra
deps beyond psutil (already in env). One sample takes ~10-50 ms (dominated by
nvidia-smi); call every N updates (not every step) so it doesn't hurt SPS.

Output:
  - CSV row appended to ``<save_dir>/sys.csv`` (one row per call)
  - Brief one-line summary string for the main training log

nvidia-smi reports ALL physical GPUs on the host regardless of
CUDA_VISIBLE_DEVICES (that env var is honored by CUDA libs, not by nvidia-smi).
That's what we want: we can spot when another workload steals util/VRAM on a
GPU we're not training on.
"""

import subprocess
from pathlib import Path

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def query_gpus():
    """Return list of dicts: {idx, mem_used_mb, mem_total_mb, util_pct, temp_c}.

    Empty list on failure (no nvidia-smi, timeout, parse error). Safe to call
    from any thread; ~10-50 ms wall-clock per call.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append({
                "idx": int(parts[0]),
                "mem_used_mb": int(parts[1]),
                "mem_total_mb": int(parts[2]),
                "util_pct": int(parts[3]),
                "temp_c": int(parts[4]),
            })
        except ValueError:
            continue
    return gpus


def query_sys():
    """Return dict of CPU/RAM stats. Zeros if psutil missing."""
    if not _HAS_PSUTIL:
        return {"cpu_pct": 0.0, "ram_used_gb": 0.0, "ram_total_gb": 0.0,
                "ram_pct": 0.0, "n_threads": 0}
    cpu_pct = psutil.cpu_percent(interval=None)  # since last call (warm up once at init)
    ram = psutil.virtual_memory()
    try:
        n_threads = psutil.Process().num_threads()
    except Exception:
        n_threads = 0
    return {
        "cpu_pct": float(cpu_pct),
        "ram_used_gb": ram.used / (1024 ** 3),
        "ram_total_gb": ram.total / (1024 ** 3),
        "ram_pct": float(ram.percent),
        "n_threads": int(n_threads),
    }


def query_all():
    return {"sys": query_sys(), "gpus": query_gpus()}


def format_brief(info):
    s = info["sys"]
    out = (f"CPU={s['cpu_pct']:>3.0f}% "
           f"RAM={s['ram_used_gb']:.1f}/{s['ram_total_gb']:.0f}G({s['ram_pct']:.0f}%) "
           f"thr={s['n_threads']}")
    for g in info["gpus"]:
        mem_pct = 100.0 * g["mem_used_mb"] / max(g["mem_total_mb"], 1)
        out += (f" | GPU{g['idx']} VRAM={g['mem_used_mb']/1024:.1f}/"
                f"{g['mem_total_mb']/1024:.0f}G({mem_pct:.0f}%) "
                f"util={g['util_pct']:>3d}% T={g['temp_c']}C")
    return out


def csv_header(n_gpus):
    h = ["upd", "cpu_pct", "ram_used_gb", "ram_total_gb", "ram_pct", "n_threads"]
    for i in range(n_gpus):
        h += [f"gpu{i}_mem_used_mb", f"gpu{i}_mem_total_mb",
              f"gpu{i}_util_pct", f"gpu{i}_temp_c"]
    return ",".join(h)


def csv_row(upd, info, n_gpus):
    s = info["sys"]
    row = [str(upd), f"{s['cpu_pct']:.1f}",
           f"{s['ram_used_gb']:.2f}", f"{s['ram_total_gb']:.2f}",
           f"{s['ram_pct']:.1f}", str(s["n_threads"])]
    # Index queried GPUs by their physical idx so columns stay aligned even
    # if nvidia-smi skips one (rare but defensive).
    by_idx = {g["idx"]: g for g in info["gpus"]}
    for i in range(n_gpus):
        g = by_idx.get(i, {"mem_used_mb": 0, "mem_total_mb": 0,
                           "util_pct": 0, "temp_c": 0})
        row += [str(g["mem_used_mb"]), str(g["mem_total_mb"]),
                str(g["util_pct"]), str(g["temp_c"])]
    return ",".join(row)


def write_csv_log(csv_path, upd, info, n_gpus):
    """Append one CSV row. Writes header on first call (file didn't exist)."""
    csv_path = Path(csv_path)
    new = not csv_path.exists()
    with open(csv_path, "a") as f:
        if new:
            f.write(csv_header(n_gpus) + "\n")
        f.write(csv_row(upd, info, n_gpus) + "\n")


if __name__ == "__main__":
    # Quick smoke test from CLI.
    import json
    info = query_all()
    print(format_brief(info))
    print(json.dumps(info, indent=2))
