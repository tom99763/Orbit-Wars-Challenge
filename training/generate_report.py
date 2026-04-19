"""Generate a single-file HTML report from AlphaZero training logs.

Reads:
  .alphazero_monitor.csv      (per-iter training stats)
  .a2c_eval_watch.csv         (optional, periodic vs-baseline win rates)

Output:
  report_<tag>.html  — self-contained, matplotlib PNGs inlined as base64

Usage:
  python training/generate_report.py --tag az_v1 \
      --monitor .alphazero_monitor.csv \
      --eval .a2c_eval_watch.csv \
      --out report_az_v1.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime
import io
import pathlib
import sys


def read_csv_rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_loss_curves(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    iters = [int(r["iter"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    labels = [("loss", "Total loss"), ("policy_loss", "Policy loss (KL)"),
              ("value_loss", "Value loss (MSE)"), ("entropy", "Entropy")]
    for ax, (key, title) in zip(axes.flat, labels):
        y = [float(r.get(key) or 0) for r in rows]
        ax.plot(iters, y, lw=1.2)
        ax.set_title(title); ax.set_xlabel("iter"); ax.grid(alpha=.3)
    fig.suptitle("Training losses per iter")
    return fig_to_b64(fig)


def plot_seat_winrates(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    iters = [int(r["iter"]) for r in rows]
    seat_series: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    for r in rows:
        counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        for entry in (r.get("seat_win_counts") or "").split(";"):
            if ":" not in entry: continue
            k, v = entry.split(":"); counts[int(k)] = int(v)
        for s in range(4):
            seat_series[s].append(counts[s])
    fig, ax = plt.subplots(figsize=(11, 4))
    for s, ys in seat_series.items():
        ax.plot(iters, ys, lw=1.2, label=f"seat {s}")
    ax.set_title("Wins per seat per iter (4 games/iter baseline)")
    ax.set_xlabel("iter"); ax.set_ylabel("wins"); ax.grid(alpha=.3); ax.legend()
    return fig_to_b64(fig)


def plot_game_stats(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    iters = [int(r["iter"]) for r in rows]
    steps = [int(r["total_steps"]) for r in rows]
    mcts = [int(r.get("mcts_calls") or 0) for r in rows]
    wall = [float(r["wall_seconds"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].plot(iters, steps, lw=1.2); axes[0].set_title("Total steps / iter (4 games)")
    axes[0].set_xlabel("iter"); axes[0].grid(alpha=.3)
    axes[1].plot(iters, mcts, lw=1.2, color="tab:orange")
    axes[1].set_title("MCTS calls / iter"); axes[1].set_xlabel("iter"); axes[1].grid(alpha=.3)
    axes[2].plot(iters, wall, lw=1.2, color="tab:green")
    axes[2].set_title("Wall seconds / iter"); axes[2].set_xlabel("iter"); axes[2].grid(alpha=.3)
    return fig_to_b64(fig)


def plot_sample_balance(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    iters = [int(r["iter"]) for r in rows]
    n_tot = [int(r["n_total"]) for r in rows]
    n_lrn = [int(r["n_learner"]) for r in rows]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(iters, n_tot, lw=1.2, label="all records (value loss)")
    ax.plot(iters, n_lrn, lw=1.2, label="learner only (policy loss)")
    ax.set_title("Training sample counts per iter")
    ax.set_xlabel("iter"); ax.set_ylabel("# records"); ax.grid(alpha=.3); ax.legend()
    return fig_to_b64(fig)


def plot_eval_rates(eval_rows):
    if not eval_rows:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = list(range(len(eval_rows)))
    starter = [float(r.get("vs_starter_wr") or 0) for r in eval_rows]
    lb928 = [float(r.get("vs_lb928_wr") or 0) for r in eval_rows]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ts, starter, marker="o", lw=1.5, label="vs starter")
    ax.plot(ts, lb928, marker="s", lw=1.5, label="vs lb-928")
    ax.axhline(0.5, color="gray", ls="--", alpha=.5)
    ax.set_title("Eval win rate over checkpoints (watcher)")
    ax.set_xlabel("checkpoint save index"); ax.set_ylabel("win rate")
    ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend()
    return fig_to_b64(fig)


def summary_table(rows):
    if not rows:
        return "<p>(no training data)</p>"
    first, last = rows[0], rows[-1]
    def f(k, last_or_first): return last_or_first.get(k) or "-"
    html = "<table><thead><tr><th>metric</th><th>iter 0</th><th>last iter</th></tr></thead><tbody>"
    for k in ["loss", "policy_loss", "value_loss", "entropy", "wall_seconds", "mcts_calls"]:
        html += f"<tr><td>{k}</td><td>{f(k, first)}</td><td>{f(k, last)}</td></tr>"
    html += f"<tr><td>total iters</td><td colspan=2>{len(rows)}</td></tr>"
    total_games = len(rows) * 4  # 4 per iter by default
    html += f"<tr><td>estimated games</td><td colspan=2>{total_games}</td></tr>"
    html += "</tbody></table>"
    return html


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="az_v1")
    ap.add_argument("--monitor", default=".alphazero_monitor.csv")
    ap.add_argument("--eval", default=".a2c_eval_watch.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    mon_rows = read_csv_rows(root / args.monitor)
    eval_rows = read_csv_rows(root / args.eval)

    out_path = pathlib.Path(args.out) if args.out else (
        root / f"report_{args.tag}.html")

    if not mon_rows:
        print(f"WARNING: no data in {args.monitor}", file=sys.stderr)

    sections = []
    sections.append(("Summary", summary_table(mon_rows)))
    if mon_rows:
        sections.append(("Loss curves", f'<img src="data:image/png;base64,{plot_loss_curves(mon_rows)}">'))
        sections.append(("Seat win distribution", f'<img src="data:image/png;base64,{plot_seat_winrates(mon_rows)}">'))
        sections.append(("Game / MCTS statistics", f'<img src="data:image/png;base64,{plot_game_stats(mon_rows)}">'))
        sections.append(("Sample counts (all-seats value vs learner-only policy)",
                         f'<img src="data:image/png;base64,{plot_sample_balance(mon_rows)}">'))
    evalplot = plot_eval_rates(eval_rows)
    if evalplot:
        sections.append(("Eval vs baselines", f'<img src="data:image/png;base64,{evalplot}">'))

    html = ["<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>AlphaZero report — {args.tag}</title>",
            "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;color:#222}",
            "h1{border-bottom:2px solid #333;padding-bottom:.3em}",
            "h2{background:#f0f0f0;padding:.4em 1em;margin-top:2em}",
            "table{border-collapse:collapse;margin:1em 0}",
            "th,td{border:1px solid #999;padding:.3em .8em;text-align:right}",
            "img{max-width:100%;height:auto;margin:1em 0;border:1px solid #ccc}",
            "</style></head><body>",
            f"<h1>AlphaZero training report — {args.tag}</h1>",
            f"<p>Generated {datetime.datetime.now().isoformat(timespec='seconds')}.</p>"]
    for title, body in sections:
        html.append(f"<h2>{title}</h2>")
        html.append(body)
    html.append("</body></html>")
    out_path.write_text("\n".join(html), encoding="utf-8")
    print(f"wrote {out_path}  ({len(mon_rows)} train iters, {len(eval_rows)} eval rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
