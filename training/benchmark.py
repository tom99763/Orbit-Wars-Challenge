"""Deterministic benchmark harness for Orbit Wars agents.

Why this exists: eval_suite.py and eval_local.py don't fix seeds, so
re-running the same checkpoint against the same opponent can give different
win rates. This makes checkpoint-to-checkpoint comparisons noisy and hides
small but real policy gains.

This harness:
  - Fixes `random.seed(seed_base + game_idx)` before each env.reset, so
    identical (me, opp, seed_base, game_idx) always reproduces the same
    game outcome.
  - Derives 2P/4P mix and starting seat deterministically from game_idx,
    so that rotating through checkpoints hits the same distribution of
    scenarios.
  - Accepts both .py files (external agents via `agent` callable) and .pt
    checkpoints (via training.agent_v4.load_agent).
  - Cross-products multiple `--me` and multiple `--opponents` into a
    win-rate matrix.
  - Appends one aggregate row per (me, opp) pairing to
    training/logs/benchmark.csv.

Usage:
  python training/benchmark.py \
      --me main.py,training/checkpoints/k14_vec_smoke_iter00048.pt \
      --opponents starter,lb928,lb1200 \
      --n-games 30 --seed-base 1000 --four-player-prob 0.2

Stdout shows a formatted matrix; CSV persists the history.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.orbit_wars.orbit_wars import (  # noqa: E402
    random_agent,
    starter_agent,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "training" / "logs" / "benchmark.csv"

CSV_COLUMNS = [
    "timestamp_utc",
    "me_label",
    "me_path",
    "opp_label",
    "opp_path",
    "n_games",
    "seed_base",
    "four_player_prob",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "mean_steps",
    "wall_seconds",
    "seat_breakdown_json",
    "note",
]


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

BUILTIN_OPPONENTS: dict[str, object] = {
    "starter": starter_agent,
    "random": random_agent,
}


def _load_lb_agent(name: str):
    """Lazily load lb928 / lb1200 (they have big top-level constants)."""
    if name == "lb928":
        from training.lb928_agent import agent as fn
        return fn
    if name == "lb1200":
        from training.lb1200_agent import agent as fn
        return fn
    raise ValueError(f"Unknown lb opponent: {name}")


def _load_py_agent(path: pathlib.Path):
    """Import an external .py file and return its `agent` callable."""
    spec = importlib.util.spec_from_file_location(
        f"bench_agent_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot spec-load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise RuntimeError(f"{path} does not export `agent`")
    return mod.agent


def _load_ckpt_agent(path: pathlib.Path, device: str, temperature: float):
    """Load a .pt checkpoint, auto-detecting the correct loader.

    - k14/k13 checkpoints (DualStreamK13Agent, saved as {"model": sd, ...}
      without a "kwargs" key) → training.k14_agent_wrapper.load_k14_agent.
    - Older OrbitAgent BC checkpoints (with ckpt["kwargs"])
      → training.agent_v4.load_agent.
    """
    from training.k14_agent_wrapper import is_k14_checkpoint, load_k14_agent
    if is_k14_checkpoint(path):
        return load_k14_agent(str(path), device=device, temperature=temperature)
    from training.agent_v4 import load_agent
    return load_agent(str(path), device=device, temperature=temperature)


def resolve_agent(spec: str, device: str, temperature: float):
    """Turn a spec string into (label, callable, path_for_logging).

    Accepts:
      - Built-in names: "starter", "random", "lb928", "lb1200"
      - Path to .py   : exec the module, use its `agent` callable
      - Path to .pt   : auto-detected k14 or OrbitAgent loader
    """
    if spec in BUILTIN_OPPONENTS:
        return spec, BUILTIN_OPPONENTS[spec], spec
    if spec in ("lb928", "lb1200"):
        return spec, _load_lb_agent(spec), spec

    path = pathlib.Path(spec)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"agent spec not found: {spec} (resolved {path})")

    if path.suffix == ".py":
        return path.stem, _load_py_agent(path), str(path)
    if path.suffix == ".pt":
        return path.stem, _load_ckpt_agent(path, device, temperature), str(path)
    raise ValueError(f"unsupported agent file type: {path.suffix}")


def expand_specs(raw: str) -> list[str]:
    """Expand a comma-separated spec string with optional glob patterns.

    Examples:
      "starter,lb928"                                → ["starter","lb928"]
      "training/checkpoints/k14_vec_smoke_iter*.pt"  → [paths...]
      "main.py,training/checkpoints/*.pt"            → [main.py, ckpts...]

    Globs that match no files raise FileNotFoundError (to fail loudly
    instead of silently producing an empty roster).
    """
    out: list[str] = []
    for item in (x.strip() for x in raw.split(",") if x.strip()):
        if any(c in item for c in "*?["):
            # Resolve glob relative to REPO_ROOT when not absolute
            pat = item
            base = REPO_ROOT
            if pathlib.Path(pat).is_absolute():
                base = pathlib.Path(pat).anchor or pathlib.Path("/")
                pat = str(pathlib.Path(pat).relative_to(base))
            matches = sorted(pathlib.Path(base).glob(pat))
            if not matches:
                raise FileNotFoundError(f"glob matched nothing: {item}")
            out.extend(str(m) for m in matches)
        else:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Game-running
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    seed: int
    n_players: int
    me_seat: int
    outcome: str  # "win" | "loss" | "draw"
    n_steps: int


@dataclass
class PairingResult:
    me_label: str
    me_path: str
    opp_label: str
    opp_path: str
    n_games: int
    seed_base: int
    four_player_prob: float
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_steps: int = 0
    wall_seconds: float = 0.0
    seat_stats: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    # seat_stats[(n_players, me_seat)] = [wins, n_games]

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_games if self.n_games else 0.0

    @property
    def mean_steps(self) -> float:
        return self.total_steps / self.n_games if self.n_games else 0.0


def _schedule(game_idx: int, four_player_prob: float) -> tuple[int, int]:
    """Deterministic (n_players, me_seat) derived from game_idx only.

    Uses an isolated `random.Random(game_idx)` so the sequence is
    reproducible without touching the global RNG that seeds the env.
    """
    r = random.Random(0xC0FFEE ^ game_idx)
    n_players = 4 if r.random() < four_player_prob else 2
    seat = r.randrange(n_players)
    return n_players, seat


def play_one_game(
    me,
    opp,
    n_players: int,
    me_seat: int,
    seed: int,
) -> GameResult:
    random.seed(seed)
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)
    agents = [me if s == me_seat else opp for s in range(n_players)]
    env.run(agents)
    final = env.steps[-1]
    my_r = final[me_seat].reward or 0
    others = [(final[s].reward or 0) for s in range(n_players) if s != me_seat]
    max_other = max(others) if others else 0
    if my_r > max_other:
        outcome = "win"
    elif my_r < max_other:
        outcome = "loss"
    else:
        outcome = "draw"
    return GameResult(
        seed=seed,
        n_players=n_players,
        me_seat=me_seat,
        outcome=outcome,
        n_steps=len(env.steps),
    )


def run_pairing(
    me, me_label, me_path,
    opp, opp_label, opp_path,
    n_games: int,
    seed_base: int,
    four_player_prob: float,
    verbose: bool = True,
) -> PairingResult:
    result = PairingResult(
        me_label=me_label, me_path=me_path,
        opp_label=opp_label, opp_path=opp_path,
        n_games=n_games, seed_base=seed_base,
        four_player_prob=four_player_prob,
    )
    t0 = time.time()
    for i in range(n_games):
        n_players, me_seat = _schedule(i, four_player_prob)
        seed = seed_base + i
        g = play_one_game(me, opp, n_players, me_seat, seed)
        if g.outcome == "win":
            result.wins += 1
        elif g.outcome == "loss":
            result.losses += 1
        else:
            result.draws += 1
        result.total_steps += g.n_steps
        key = (g.n_players, g.me_seat)
        result.seat_stats.setdefault(key, [0, 0])
        if g.outcome == "win":
            result.seat_stats[key][0] += 1
        result.seat_stats[key][1] += 1
        if verbose and (i + 1) % 10 == 0:
            print(
                f"  [{me_label} vs {opp_label}] {i+1}/{n_games} "
                f"W={result.wins} L={result.losses} D={result.draws}",
                flush=True,
            )
    result.wall_seconds = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# CSV & reporting
# ---------------------------------------------------------------------------

def append_csv(results: list[PairingResult], note: str) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with CSV_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_COLUMNS)
        for r in results:
            seat_json = json.dumps(
                {f"{k[0]}P_seat{k[1]}": v for k, v in sorted(r.seat_stats.items())}
            )
            w.writerow([
                ts,
                r.me_label, r.me_path,
                r.opp_label, r.opp_path,
                r.n_games, r.seed_base, r.four_player_prob,
                r.wins, r.losses, r.draws,
                f"{r.win_rate:.4f}",
                f"{r.mean_steps:.1f}",
                f"{r.wall_seconds:.1f}",
                seat_json,
                note,
            ])


# ---------------------------------------------------------------------------
# ELO from pairing results
# ---------------------------------------------------------------------------

def compute_elo(
    results: list[PairingResult],
    k_factor: float = 32.0,
    initial: float = 1200.0,
    n_epochs: int = 10,
) -> dict[str, float]:
    """Iterative ELO fit from pairwise win rates.

    We only see aggregate (wins, losses, draws) per pairing, so we replay
    each game as a single virtual match with outcome = empirical win rate:
    expanding draws into 0.5-scored matches. Multiple epochs over the same
    match list converge the ratings regardless of order sensitivity.

    Returns {label → rating}. Labels are pulled from me_label/opp_label —
    identical labels across different "me" and "opp" positions merge, so
    a checkpoint appearing on both sides of the matrix gets one rating.
    """
    labels: set[str] = set()
    for r in results:
        labels.add(r.me_label)
        labels.add(r.opp_label)
    rating = {lbl: float(initial) for lbl in labels}

    # Virtual matches: for each pairing, expand into n_games individual
    # match outcomes so draws get 0.5 weight and ordering is explicit.
    matches: list[tuple[str, str, float]] = []
    for r in results:
        for _ in range(r.wins):
            matches.append((r.me_label, r.opp_label, 1.0))
        for _ in range(r.losses):
            matches.append((r.me_label, r.opp_label, 0.0))
        for _ in range(r.draws):
            matches.append((r.me_label, r.opp_label, 0.5))

    rng = random.Random(0)
    for _ in range(n_epochs):
        rng.shuffle(matches)
        for a, b, score_a in matches:
            ra, rb = rating[a], rating[b]
            exp_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            rating[a] = ra + k_factor * (score_a - exp_a)
            rating[b] = rb + k_factor * ((1.0 - score_a) - (1.0 - exp_a))
    return rating


def print_elo(rating: dict[str, float]) -> None:
    print("\n=== ELO (K=32, initial=1200, 10 epochs) ===")
    ranked = sorted(rating.items(), key=lambda kv: kv[1], reverse=True)
    name_w = max(10, max((len(n) for n, _ in ranked), default=10))
    for rank, (name, r) in enumerate(ranked, 1):
        print(f"  {rank:>2}. {name:<{name_w}}  {r:7.1f}")


def print_matrix(results: list[PairingResult]) -> None:
    me_labels = []
    opp_labels = []
    for r in results:
        if r.me_label not in me_labels:
            me_labels.append(r.me_label)
        if r.opp_label not in opp_labels:
            opp_labels.append(r.opp_label)

    lookup = {(r.me_label, r.opp_label): r for r in results}

    print("\n=== win-rate matrix ===")
    col_w = max(12, max((len(x) for x in opp_labels), default=8) + 2)
    row_w = max(10, max((len(x) for x in me_labels), default=8) + 2)
    header = " " * row_w + "".join(f"{o:>{col_w}}" for o in opp_labels)
    print(header)
    for m in me_labels:
        cells = []
        for o in opp_labels:
            r = lookup.get((m, o))
            if r is None:
                cells.append("   —   ")
            else:
                cells.append(f"{r.wins}/{r.n_games} ({r.win_rate:.0%})")
        print(f"{m:<{row_w}}" + "".join(f"{c:>{col_w}}" for c in cells))

    print("\n=== seat breakdown ===")
    for r in results:
        print(f"  {r.me_label} vs {r.opp_label}:")
        for k in sorted(r.seat_stats.keys()):
            w, n = r.seat_stats[k]
            print(f"    {k[0]}P seat {k[1]}: {w}/{n}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--me", default=None,
        help="Comma-separated specs (supports globs, e.g. 'x/*.pt'). Each is "
             "a path to .py, path to .pt, or a builtin name "
             "(starter/random/lb928/lb1200).",
    )
    ap.add_argument(
        "--opponents", default="starter,lb928,lb1200",
        help="Comma-separated opponent specs (same format as --me).",
    )
    ap.add_argument(
        "--league", default=None,
        help="Round-robin shortcut: one comma-separated spec list (supports "
             "globs) evaluated as both --me and --opponents. ELO rating "
             "fitted from the resulting matrix. Mutually exclusive with --me.",
    )
    ap.add_argument("--n-games", type=int, default=30)
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="seed[i] = seed_base + i, deterministic across runs")
    ap.add_argument("--four-player-prob", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--note", default="",
                    help="Free-form note written to CSV (e.g. 'post-iter1020').")
    ap.add_argument("--no-csv", action="store_true",
                    help="Skip CSV append (dry-run).")
    ap.add_argument("--skip-self", action="store_true",
                    help="In round-robin, skip pairings where me == opp.")
    args = ap.parse_args()

    if (args.me is None) == (args.league is None):
        ap.error("exactly one of --me or --league must be provided")

    if args.league is not None:
        expanded = expand_specs(args.league)
        me_specs = list(expanded)
        opp_specs = list(expanded)
    else:
        me_specs = expand_specs(args.me)
        opp_specs = expand_specs(args.opponents)

    print(f"loading {len(me_specs)} me-agent(s) and {len(opp_specs)} opponent(s)",
          flush=True)
    mes = [resolve_agent(s, args.device, args.temperature) for s in me_specs]
    opps = [resolve_agent(s, args.device, args.temperature) for s in opp_specs]

    total_pairings = len(mes) * len(opps)
    if args.skip_self:
        total_pairings -= sum(
            1 for (ml, _, _) in mes for (ol, _, _) in opps if ml == ol
        )
    print(f"plan: {len(mes)}×{len(opps)} = {total_pairings} pairings, "
          f"{args.n_games} games each, seed_base={args.seed_base}, "
          f"4P_prob={args.four_player_prob}", flush=True)

    all_results: list[PairingResult] = []
    t0 = time.time()
    for me_label, me_fn, me_path in mes:
        for opp_label, opp_fn, opp_path in opps:
            if args.skip_self and me_label == opp_label:
                continue
            print(f"\n[{me_label} vs {opp_label}] starting...", flush=True)
            r = run_pairing(
                me_fn, me_label, me_path,
                opp_fn, opp_label, opp_path,
                n_games=args.n_games,
                seed_base=args.seed_base,
                four_player_prob=args.four_player_prob,
            )
            print(
                f"  → {r.wins}W/{r.losses}L/{r.draws}D "
                f"({r.win_rate:.1%})  [{r.wall_seconds:.0f}s, "
                f"mean_steps={r.mean_steps:.0f}]",
                flush=True,
            )
            all_results.append(r)

    print_matrix(all_results)

    # Compute + print ELO whenever there's more than one distinct agent
    # involved (otherwise ratings are trivially the initial value).
    distinct_agents = {r.me_label for r in all_results} | {
        r.opp_label for r in all_results
    }
    if len(distinct_agents) > 1:
        rating = compute_elo(all_results)
        print_elo(rating)

    if not args.no_csv:
        append_csv(all_results, note=args.note)
        print(f"\nappended {len(all_results)} rows to {CSV_PATH}")
    print(f"\ntotal wall: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
