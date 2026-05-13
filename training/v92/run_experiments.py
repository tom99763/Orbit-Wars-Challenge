"""Autonomous experiment runner — runs baseline + all inductive-bias variants sequentially.

For each experiment:
  1. Train ~1500 upd with the specified bias config
  2. Eval against starter/v14/lb1224/ow_proto at upd 500/1000/1500
  3. Log metrics to per-experiment save_dir
  4. Move on to next bias

At end, run `make_html_report.py` to summarize.

Each experiment writes to: save/v92_exp/{name}/
"""
import os, sys, time, subprocess, signal, argparse, json
sys.path.insert(0, "/home/lab/orbit-war")
from pathlib import Path
from typing import Dict, Optional


# Each experiment = a flag dict written as env vars (train_jax reads them).
# Baseline already includes: sun-block, ship-availability, K=7 nearest, hardcoded
# sniper ship count. One-delta-at-a-time: queue ONLY baseline now; uncomment
# the next variant only AFTER confirming baseline reaches some non-trivial WR
# (otherwise we'd compound action-space changes onto an unstable base).
_SNAP = {}  # No env-flag needed — PFSP pool is always-on inside train_jax (kicks in at upd 600)
def _merge(d, extra=_SNAP):
    out = dict(d); out.update(extra); return out

# All variants run with PFSP-pool self-play (pool init upd 600, snap every
# 50 upd, max 20 entries; PFSP weights ∝ (1 − learner_WR)²) and inline eval
# every 200 upd. Both are hardcoded in train_jax.py so baseline inherits them.
# Baseline re-added 2026-05-13 after R20 + audit fixes shifted training
# trajectory — variants need a clean apples-to-apples reference.
EXPERIMENTS = [
    # Reference run — zero inductive biases, default K_NEAREST=7, sniper ships
    ("baseline",        "no biases — K=7 nearest, sniper ship rule, PFSP pool",         _merge({})),
    # Tier 0 — action-space (ow_proto-derived, all coded)
    ("AS1_K_4",         "K=4 nearest candidates (narrower)",                            _merge({"K_NEAREST": "4"})),
    ("AS1_K_12",        "K=12 nearest candidates (wider)",                              _merge({"K_NEAREST": "12"})),
    ("AS2_req_ships",   "ow_proto req_ships: ships = base + tgt.prod × eta",            _merge({"BIAS_REQ_SHIPS": "1"})),
    ("AS3_ship_head",   "learnable ship_head (7 buckets)",                              _merge({"USE_SHIP_HEAD": "1"})),
    ("AS4_intercept",   "3-iter intercept solve (moving inner planets)",                _merge({"BIAS_INTERCEPT": "1"})),
    ("AS5_score_cand",  "rank candidates by ow_proto custom_score",                     _merge({"BIAS_SCORE_CAND": "1"})),
    ("AS6_hostility",   "candidate quota: 3 enemy + 2 neutral + 2 friendly",            _merge({"BIAS_HOSTILITY_QUOTAS": "1"})),
    ("AS8_doomed",      "mask targets already-doomed by incoming enemy fleets",         _merge({"BIAS_DOOMED_FILTER": "1"})),
    # Tier 1 — observation features
    ("T2_capture_cost", "capture_cost = enemy_ships + prod×eta",                        _merge({"BIAS_CAPTURE_COST": "1"})),
    ("T2_threat_inflow","threat_inflow per planet",                                     _merge({"BIAS_THREAT_INFLOW": "1"})),
    ("T2_prod_cp",      "prod/capture_cost CP-value",                                   _merge({"BIAS_PROD_CP": "1", "BIAS_CAPTURE_COST": "1"})),
    # Tier 2 — advanced features
    ("A1_defense_look", "defense_lookahead 28-turn",                                    _merge({"BIAS_DEFENSE_LOOK": "1"})),
    ("A2_crash_exploit","crash_exploit (incoming ≤ 5 turns)",                           _merge({"BIAS_CRASH_EXPLOIT": "1"})),
    ("A3_gang_up",      "gang_up (4-turn enemy stack window)",                          _merge({"BIAS_GANG_UP": "1"})),
    # Tier 3 — global / phase features
    ("S1_stage_onehot", "stage one-hot (early/mid/late)",                               _merge({"BIAS_STAGE_ONEHOT": "1"})),
    ("S3_sundancer",    "sundancer 3-phase (EXPAND/DOMINATE/BAIT)",                     _merge({"BIAS_SUNDANCER_PHASE": "1"})),
    # Tier 4 — training-level
    ("T3_pbrs",         "PBRS shaping Φ=(my-en)/200 (Ng 1999)",                         _merge({"BIAS_PBRS": "1"})),
]


def _train_proc(save_dir: Path, env_extras: Dict[str, str],
                n_envs: int, t_rollout: int, total_updates: int):
    env = dict(os.environ)
    env["PYTHONPATH"] = "/home/lab/orbit-war"
    env.update(env_extras)
    cmd = [
        sys.executable, "training/v92/train_jax.py",
        "--n-envs", str(n_envs),
        "--t-rollout", str(t_rollout),
        "--total-updates", str(total_updates),
        "--save-dir", str(save_dir),
    ]
    log_path = save_dir / "run.log"
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                  cwd="/home/lab/orbit-war")
    return proc


def _watcher_proc(save_dir: Path):
    log_path = save_dir / "eval_watcher.log"
    cmd = [
        "/tmp/run_watcher_wrap.sh", str(save_dir),
    ]
    # Generate wrapper script once
    wrap_path = Path("/tmp/run_watcher_wrap.sh")
    if not wrap_path.exists():
        wrap_path.write_text(
            "#!/bin/bash\n"
            "source ~/miniconda3/etc/profile.d/conda.sh\n"
            "conda activate tom\n"
            "export PYTHONPATH=/home/lab/orbit-war\n"
            "export JAX_PLATFORMS=cpu\n"
            'exec python /home/lab/orbit-war/training/v92/eval_watcher.py --save-dir "$1"\n'
        )
        wrap_path.chmod(0o755)
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc


def _is_stable(csv_path: Path, window: int = 50) -> Optional[Dict]:
    """Check training stability: ent_t > 1.0, ent_s > 0.5, clip_frac < 0.20 over last N upds."""
    import pandas as pd
    if not csv_path.exists(): return None
    df = pd.read_csv(csv_path)
    if len(df) < window: return None
    recent = df.tail(window)
    ok_ent_t = (recent["ent_t"] > 1.0).all()
    ok_ent_s = (recent["ent_s"] > 0.5).all()
    ok_cf = (recent["clip_frac"] < 0.20).all()
    return {
        "stable": ok_ent_t and ok_ent_s and ok_cf,
        "ent_t_mean": float(recent["ent_t"].mean()),
        "ent_s_mean": float(recent["ent_s"].mean()),
        "cf_mean": float(recent["clip_frac"].mean()),
        "upd": int(recent["upd"].iloc[-1]),
    }


def run_experiment(name: str, description: str, env_extras: Dict[str, str],
                    base_save: Path, n_envs: int, t_rollout: int,
                    total_updates: int, max_wall_sec: int = 3600):
    """Run one experiment. Returns dict with metrics."""
    save_dir = base_save / name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "snapshots").mkdir(exist_ok=True)
    print(f"\n{'='*60}\n[exp] {name}: {description}\n{'='*60}", flush=True)
    print(f"  env_extras={env_extras}", flush=True)

    train_proc = _train_proc(save_dir, env_extras, n_envs, t_rollout, total_updates)
    print(f"  train PID {train_proc.pid} (eval inline every 200 upd inside train_jax)", flush=True)

    csv_path = save_dir / "train.csv"
    eval_csv = save_dir / "eval.csv"
    t_start = time.time()
    last_log = 0
    try:
        while True:
            time.sleep(30)
            wall = time.time() - t_start
            # Per-experiment cf trajectory check (practitioner's #1 warning)
            if csv_path.exists():
                import pandas as pd
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    df = pd.DataFrame()
                if len(df) > 100 and "clip_frac" in df.columns and "ent_t" in df.columns:
                    recent_cf = df["clip_frac"].tail(50).mean()
                    earlier_cf = df["clip_frac"].iloc[-100:-50].mean()
                    if recent_cf > 0.25 and recent_cf > earlier_cf * 1.5:
                        print(f"  ⚠️ {name}: clip_frac MONOTONIC RISE ({earlier_cf:.3f}→{recent_cf:.3f}) — early kill", flush=True)
                        train_proc.terminate()
                        try: train_proc.wait(timeout=30)
                        except subprocess.TimeoutExpired: train_proc.kill()
                        break
                    # Detect entropy collapse
                    if len(df) > 50:
                        recent_ent_t = df["ent_t"].tail(50).mean()
                        if recent_ent_t < 0.3:
                            print(f"  ⚠️ {name}: ent_t collapsed ({recent_ent_t:.3f}) — early kill", flush=True)
                            train_proc.terminate()
                            try: train_proc.wait(timeout=30)
                            except subprocess.TimeoutExpired: train_proc.kill()
                            break
            if wall - last_log > 120:
                if csv_path.exists():
                    import pandas as pd
                    try:
                        df = pd.read_csv(csv_path)
                        if len(df) > 0 and "upd" in df.columns:
                            last = df.iloc[-1]
                            print(f"  [{name} t={wall:.0f}s upd={int(last['upd'])}] "
                                  f"ent_t={last['ent_t']:.2f} ent_s={last['ent_s']:.2f} "
                                  f"cf={last['clip_frac']:.3f} sps={last['sps']:.0f}",
                                  flush=True)
                        else:
                            print(f"  [{name} t={wall:.0f}s] train.csv has no rows yet "
                                  f"or missing 'upd' column", flush=True)
                    except Exception as e:
                        print(f"  [{name} t={wall:.0f}s] train.csv read failed: "
                              f"{type(e).__name__}: {e!s:.80}", flush=True)
                last_log = wall
            if train_proc.poll() is not None:
                print(f"  train exited code {train_proc.returncode} after {wall:.0f}s", flush=True)
                break
            if wall > max_wall_sec:
                print(f"  wall limit {max_wall_sec}s hit, terminating", flush=True)
                train_proc.terminate()
                try: train_proc.wait(timeout=30)
                except subprocess.TimeoutExpired: train_proc.kill()
                break
    finally:
        pass  # no watcher — eval is inline in train_jax

    # ── Final eval on last snapshot (in case inline missed the last upd) ──
    print(f"  [{name}] training done — running final eval on last snapshot…", flush=True)
    snap_dir = save_dir / "snapshots"
    snaps = sorted(snap_dir.glob("v92_jax_upd*.pkl")) if snap_dir.exists() else []
    if snaps:
        try:
            import jax.random as jr_
            import pickle as _pkl
            from training.v92.policy_jax import init_policy as _init_policy
            from training.v92.eval_batched import eval_snapshot as _eval_snapshot
            with open(snaps[-1], "rb") as f:
                ck = _pkl.load(f)
            body_, th_, sh_, body_p_, th_p_, sh_p_ = _init_policy(jr_.PRNGKey(0))
            body_p_ = ck["body"]; th_p_ = ck["th"]; sh_p_ = ck["sh"]
            results_eval = _eval_snapshot(
                body_, th_, sh_, body_p_, th_p_, sh_p_,
                opponents=["starter", "v14", "lb1224", "ow_proto"],
                n_games_per_opp=8, base_seed=99001,
            )
            # Write to eval.csv with same header as watcher would
            opps = ["starter", "v14", "lb1224", "ow_proto"]
            ev_csv = save_dir / "eval.csv"
            if not ev_csv.exists():
                with open(ev_csv, "w") as f:
                    f.write("upd," + ",".join(f"wr_{o}" for o in opps) + ","
                            + ",".join(f"wr_sample_{o}" for o in opps) + ","
                            + ",".join(f"s0_{o}" for o in opps) + ","
                            + ",".join(f"s1_{o}" for o in opps) + ","
                            + ",".join(f"s0_sample_{o}" for o in opps) + ","
                            + ",".join(f"s1_sample_{o}" for o in opps) + ",wall_sec\n")
            upd_n = int(ck.get("upd", 0))
            with open(ev_csv, "a") as f:
                row = [str(upd_n)]
                row += [f"{results_eval[o].get('wr', 0):.3f}" for o in opps]
                row += [f"{results_eval[o].get('wr_sample', 0):.3f}" for o in opps]
                row += [str(results_eval[o].get('s0', 0)) for o in opps]
                row += [str(results_eval[o].get('s1', 0)) for o in opps]
                row += [str(results_eval[o].get('s0_sample', 0)) for o in opps]
                row += [str(results_eval[o].get('s1_sample', 0)) for o in opps]
                row.append(f"{time.time() - t_start:.1f}")
                f.write(",".join(row) + "\n")
        except Exception as e:
            print(f"  [{name}] inline eval failed: {e}", flush=True)

    # Collect final metrics
    result = {"name": name, "description": description, "env_extras": env_extras,
              "wall_sec": time.time() - t_start, "save_dir": str(save_dir)}
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            result["final_upd"] = int(df["upd"].iloc[-1])
            result["final_sps"] = float(df["sps"].iloc[-1])
            tail = df.tail(50)
            result["ent_t_mean_last50"] = float(tail["ent_t"].mean())
            result["ent_s_mean_last50"] = float(tail["ent_s"].mean())
            result["clip_frac_mean_last50"] = float(tail["clip_frac"].mean())
    if eval_csv.exists():
        import pandas as pd
        edf = pd.read_csv(eval_csv)
        if len(edf) > 0:
            result["eval_history"] = edf.to_dict("records")
            last = edf.iloc[-1]
            for col in ["wr_starter", "wr_v14", "wr_lb1224", "wr_ow_proto"]:
                if col in edf.columns:
                    result[f"final_{col}"] = float(last[col])
    # Save per-exp result json
    with open(save_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-save", default="save/v92_exp")
    ap.add_argument("--n-envs", type=int, default=128)
    ap.add_argument("--t-rollout", type=int, default=64)    # current settings (was 32)
    ap.add_argument("--updates-per-exp", type=int, default=1200)
    ap.add_argument("--baseline-updates", type=int, default=2500,
                    help="Baseline runs longer to confirm stability")
    ap.add_argument("--max-wall-sec", type=int, default=4500,
                    help="Max wall time per experiment (~75 min)")
    ap.add_argument("--skip", nargs="*", default=[], help="Skip these experiment names")
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="Names of experiments to run. Empty = all sequentially. "
                         "Baseline-stability gate only enforced when baseline is in the list "
                         "and runs first.")
    args = ap.parse_args()

    base = Path(args.base_save)
    base.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Filter EXPERIMENTS by --experiments / --skip
    selected = EXPERIMENTS
    if args.experiments:
        names_wanted = set(args.experiments)
        unknown = names_wanted - {n for n, _, _ in EXPERIMENTS}
        if unknown:
            print(f"[runner] WARN: unknown experiment names {unknown}", flush=True)
        selected = [(n, d, e) for n, d, e in EXPERIMENTS if n in names_wanted]
    if args.skip:
        selected = [(n, d, e) for n, d, e in selected if n not in args.skip]
    print(f"[runner] Running {len(selected)} experiment(s): {[n for n, _, _ in selected]}", flush=True)
    if not selected:
        print("[runner] No experiments selected. Exiting.", flush=True)
        return

    # ── Phase 1: baseline gate (only if 'baseline' is selected and first) ──
    has_baseline_gate = selected[0][0] == "baseline"
    if has_baseline_gate:
        baseline_name, baseline_desc, baseline_env = selected[0]
        print(f"\n[runner] Phase 1 — baseline stability gate", flush=True)
        result = run_experiment(baseline_name, baseline_desc, baseline_env, base,
                                 args.n_envs, args.t_rollout, args.baseline_updates,
                                 args.max_wall_sec)
        all_results.append(result)
    else:
        # No baseline gate — go straight to variant loop
        baseline_name = None

    # Check baseline stability (only if baseline was actually run)
    stable = False
    if has_baseline_gate:
        baseline_csv = base / baseline_name / "train.csv"
        if baseline_csv.exists():
            import pandas as pd
            df = pd.read_csv(baseline_csv)
            if len(df) >= 200:
                tail = df.tail(100)
                ok_ent_t = (tail["ent_t"] > 1.0).all()
                ok_ent_s = (tail["ent_s"] > 0.5).all()
                ok_cf = (tail["clip_frac"] < 0.20).all()
                stable = ok_ent_t and ok_ent_s and ok_cf
                print(f"[runner] Baseline stability check (last 100 upd):", flush=True)
                print(f"    ent_t > 1.0: {ok_ent_t} (mean {tail['ent_t'].mean():.3f})", flush=True)
                print(f"    ent_s > 0.5: {ok_ent_s} (mean {tail['ent_s'].mean():.3f})", flush=True)
                print(f"    cf < 0.20:  {ok_cf} (mean {tail['clip_frac'].mean():.3f})", flush=True)
            else:
                print(f"[runner] Baseline only {len(df)} upds — too short for stability check", flush=True)

    # Baseline gate check: only halt if baseline was actually run AND failed
    if has_baseline_gate and not stable:
        print(f"\n[runner] ⚠️ Baseline NOT stable — HALTING variant experiments", flush=True)
        print(f"  Generating HTML report with baseline only", flush=True)
        with open(base / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        try:
            subprocess.run([sys.executable, "training/v92/make_html_report.py",
                             "--results", str(base / "all_results.json"),
                             "--out", str(base / "report.html")],
                            cwd="/home/lab/orbit-war", check=True)
        except Exception:
            pass
        return

    # ── Phase 2: variants (or all-of-`selected` if no baseline gate) ──
    if has_baseline_gate:
        print(f"\n[runner] ✓ Baseline stable — proceeding to {len(selected)-1} variants", flush=True)
        rest = selected[1:]
    else:
        print(f"\n[runner] Running {len(selected)} selected experiment(s) — no baseline gate", flush=True)
        rest = selected
    for name, desc, env_extras in rest:
        if name in args.skip:
            print(f"[skip] {name}", flush=True)
            continue
        try:
            result = run_experiment(name, desc, env_extras, base, args.n_envs,
                                     args.t_rollout, args.updates_per_exp,
                                     args.max_wall_sec)
            all_results.append(result)
        except Exception as e:
            print(f"[runner] {name} CRASHED: {e}", flush=True)
            all_results.append({"name": name, "description": desc, "err": str(e)})
        time.sleep(10)

    # Save aggregate
    with open(base / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[runner] DONE — wrote {base}/all_results.json", flush=True)

    # Generate HTML report
    try:
        subprocess.run([sys.executable, "training/v92/make_html_report.py",
                         "--results", str(base / "all_results.json"),
                         "--out", str(base / "report.html")],
                        cwd="/home/lab/orbit-war", check=True)
        print(f"[runner] HTML report: {base}/report.html", flush=True)
    except Exception as e:
        print(f"[runner] HTML gen failed: {e}", flush=True)


if __name__ == "__main__":
    main()
