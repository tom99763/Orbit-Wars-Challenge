"""Per-episode match-quality and per-agent advanced stats.

Consumes `trajectories/<date>/*.pkl` + `index.csv` and emits:

  processed/<date>/analytics_agent.csv   — one row per (episode, agent)
  processed/<date>/analytics_match.csv   — one row per episode
  processed/<date>/win_prob_model.pkl    — fitted logistic classifier

Design rationale in `wiki/match-quality.md` + memory entry
`analytics_plan.md`. Metrics implemented:
  • GEI — Σ|ΔP(win)| per step, normalised by episode length
  • WPA — per-agent, per-turn attribution of ΔP(win)
  • Comeback magnitude — max_deficit(winner) - final_margin
  • Num lead changes — flips in argmax P(win) after turn 20
  • Decisive turn — first turn winner's P(win) ≥ 0.9 and never dips
  • Quality tag — epic / comeback / close / standard / blowout
  • Per-agent: attack_eff, conquest_count, defense_rate, reinforce_speed,
    comet_grabs, sun_kills, overcommit_rate, clutch_wpa_from_behind

P(win) model (v1): logistic regression on
  [ship_lead, planet_lead, production_lead, step_norm, is_4p]
fit on every (step, agent) across the supplied trajectories.
Self-supervised — label = `agent was the eventual winner` (±1 or 1/0).
"""

import argparse
import csv
import math
import pathlib
import pickle
import sys
from collections import defaultdict

import numpy as np


# -----------------------------------------------------------
# Feature extraction for the P(win) model
# -----------------------------------------------------------

def step_features(step, agent_idx):
    """Return the (x_scalar_feature_vector, … ) for one (step, agent).
    Mirrors what we'll use for both training the logistic and
    evaluating P(win_t | state_t)."""
    my_ships = step["my_total_ships"]
    en_ships = step["enemy_total_ships"]
    my_pl = step["my_planet_count"]
    en_pl = step["enemy_planet_count"]
    planets = step["planets"]
    my_prod = sum(p[6] for p in planets if p[1] == agent_idx)
    en_prod = sum(p[6] for p in planets if p[1] != agent_idx and p[1] != -1)
    total_ships = max(1, my_ships + en_ships)
    total_planets = max(1, my_pl + en_pl + step["neutral_planet_count"])
    return np.array([
        (my_ships - en_ships) / total_ships,
        (my_pl - en_pl) / total_planets,
        my_prod - en_prod,
        step["step"] / 500.0,
    ], dtype=np.float64)


def fit_win_prob_model(trajs, agent_is_4p_feature=True):
    """Fit a per-step logistic `P(win | state) = σ(β·x + b)`.
    x = [ship_lead_norm, planet_lead_norm, prod_lead, step_norm, is_4p]
    label = `eventual winner for this agent`.
    Numpy closed-form IRLS — we don't want sklearn as a dep.
    """
    X_rows, y_rows = [], []
    for t in trajs:
        is_4p = 1.0 if t["n_players"] == 4 else 0.0
        label = 1.0 if t["winner"] else 0.0
        for step in t["steps"]:
            if step["step"] < 2 or step["done"]:
                continue
            feats = step_features(step, t["agent_idx"])
            row = np.concatenate([feats, [is_4p]]) if agent_is_4p_feature else feats
            X_rows.append(row)
            y_rows.append(label)
    if not X_rows:
        return None
    X = np.array(X_rows)
    y = np.array(y_rows)
    X = np.hstack([X, np.ones((X.shape[0], 1))])  # bias column
    # IRLS
    beta = np.zeros(X.shape[1])
    for _ in range(25):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = p * (1 - p) + 1e-6
        grad = X.T @ (y - p)
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H + 1e-4 * np.eye(H.shape[0]), grad)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.linalg.norm(step) < 1e-6:
            break
    return beta  # last entry is bias


def predict_win_prob(beta, feats, is_4p):
    x = np.concatenate([feats, [is_4p, 1.0]])
    z = float(x @ beta)
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


# -----------------------------------------------------------
# Per-episode analytics
# -----------------------------------------------------------

def angle_target_planet(src, angle, planets):
    sx, sy = src[2], src[3]
    dx, dy = math.cos(angle), math.sin(angle)
    best_i, best = None, float("inf")
    for i, p in enumerate(planets):
        if p[0] == src[0]:
            continue
        vx, vy = p[2] - sx, p[3] - sy
        fwd = vx * dx + vy * dy
        if fwd <= 0:
            continue
        perp = abs(vx * (-dy) + vy * dx)
        if perp > p[4] + 2.0:
            continue
        score = fwd + 5.0 * perp
        if score < best:
            best = score
            best_i = i
    return best_i


def analyze_episode(trajs_for_episode, beta):
    """trajs_for_episode = list of per-agent trajectory dicts (same episode).
    All of them share `steps` but vary in `agent_idx` and `winner`.
    """
    any_t = trajs_for_episode[0]
    n_players = any_t["n_players"]
    n_steps = any_t["n_steps"]
    is_4p = 1.0 if n_players == 4 else 0.0
    winner_idx = next((t["agent_idx"] for t in trajs_for_episode if t["winner"]), None)

    # P(win_t, agent) table: [T, n_players]
    P = np.zeros((n_steps, n_players))
    for t in trajs_for_episode:
        ai = t["agent_idx"]
        for step in t["steps"]:
            i = step["step"]
            if i >= n_steps:
                continue
            if step["done"] or step["step"] < 2:
                P[i, ai] = 1.0 if t["winner"] else 0.0
            else:
                feats = step_features(step, ai)
                P[i, ai] = predict_win_prob(beta, feats, is_4p)
    # Normalise row-wise so P sums to 1 across agents
    row_sum = P.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-6] = 1.0
    P = P / row_sum

    # GEI, lead changes, decisive turn (all w.r.t. the eventual winner)
    if winner_idx is None:
        gei = 0.0
        num_lead_changes = 0
        decisive_turn = None
        comeback_magnitude = 0.0
    else:
        pwin = P[:, winner_idx]
        delta = np.abs(np.diff(pwin))
        gei = float(delta.sum()) / max(1, len(delta) / 500.0)
        # Lead changes: argmax(P) flips after step 20
        leader = P[20:].argmax(axis=1)
        num_lead_changes = int((leader[1:] != leader[:-1]).sum())
        # Decisive turn: first t after which pwin stays >= 0.9
        decisive_turn = None
        for t in range(len(pwin)):
            if pwin[t] >= 0.9 and pwin[t:].min() >= 0.9:
                decisive_turn = int(t)
                break
        # Comeback magnitude: winner's deepest deficit in ship lead
        winner_traj = trajs_for_episode[winner_idx] if winner_idx < len(trajs_for_episode) \
            else next(t for t in trajs_for_episode if t["agent_idx"] == winner_idx)
        deficits = []
        for step in winner_traj["steps"]:
            if step["step"] < 5 or step["done"]:
                continue
            deficits.append(step["my_total_ships"] - step["enemy_total_ships"])
        if deficits:
            max_deficit = min(deficits)
            final_margin = deficits[-1] if deficits else 0
            comeback_magnitude = float(max(0, final_margin - max_deficit))
        else:
            comeback_magnitude = 0.0

    total_ships_end = sum(
        max(0, t["steps"][-1]["my_total_ships"]) for t in trajs_for_episode
    )
    final_margin_rel = 0.0
    if winner_idx is not None:
        winner_traj = next(t for t in trajs_for_episode if t["agent_idx"] == winner_idx)
        wf = winner_traj["steps"][-1]
        final_margin_rel = (wf["my_total_ships"] - wf["enemy_total_ships"]) / max(1, total_ships_end)

    # Quality tag
    if comeback_magnitude >= 30 and num_lead_changes >= 3:
        tag = "epic"
    elif decisive_turn is not None and decisive_turn <= 150 and gei < 2.0:
        tag = "blowout"
    elif comeback_magnitude >= 20:
        tag = "comeback"
    elif abs(final_margin_rel) <= 0.1:
        tag = "close"
    else:
        tag = "standard"

    # Per-agent stats
    agent_rows = []
    for t in trajs_for_episode:
        ai = t["agent_idx"]
        steps = t["steps"]

        # Offense
        attack_ships = 0
        conquests = 0
        planets_ever_owned = set()
        # track (fleet_id, ships) → came_from_contested?
        my_final_planets = {p[0] for p in steps[-1]["planets"] if p[1] == ai}
        for i in range(1, len(steps)):
            cur_owned = {p[0] for p in steps[i]["planets"] if p[1] == ai}
            planets_ever_owned |= cur_owned
            prev_owned = {p[0] for p in steps[i-1]["planets"] if p[1] == ai}
            gained = cur_owned - prev_owned
            if gained:
                for pid in gained:
                    if pid in my_final_planets:
                        conquests += 1
            attack_ships += sum(m[2] for m in (steps[i]["action"] or []))

        # Defense
        incoming_defended = 0
        incoming_total = 0
        reinforce_latencies = []
        for i in range(1, len(steps) - 1):
            my_planets = {p[0]: p for p in steps[i]["planets"] if p[1] == ai}
            if not my_planets:
                continue
            # Incoming = enemy fleets whose predicted destination (by angle)
            # is one of my planets, within 30 units
            fleets = steps[i]["fleets"]
            targets_incoming = set()
            for f in fleets:
                if f[1] == ai:
                    continue
                # Distance to each of my planets
                for pid, p in my_planets.items():
                    d = math.hypot(p[2] - f[2], p[3] - f[3])
                    if d < 30:
                        targets_incoming.add(pid)
                        break
            incoming_total += len(targets_incoming)
            # Kept = planet still mine next step AND garrison not catastrophic
            for pid in targets_incoming:
                nxt = next((p for p in steps[i+1]["planets"] if p[0] == pid), None)
                if nxt and nxt[1] == ai:
                    incoming_defended += 1
            # Reinforcement: did we launch from my_planets this step?
            for move in (steps[i]["action"] or []):
                if move[0] in targets_incoming:
                    reinforce_latencies.append(0)

        # Comet grabs
        comet_grabs = 0
        init_ids = {p[0] for p in t["initial_planets"] or []}
        seen_comets = set()
        for step in steps:
            for p in step["planets"]:
                if p[0] not in init_ids and p[1] == ai and p[0] not in seen_comets:
                    seen_comets.add(p[0])
                    comet_grabs += 1

        # Sun kills — fleets that disappeared near the sun (point-to-segment
        # detection done by env; we approximate: fleet of mine exists at t
        # but isn't at t+1 AND no enemy-planet captured by me in the meantime
        # AND fleet was within 15 of sun at t).
        sun_kills = 0
        for i in range(len(steps) - 1):
            my_fl_t = {f[0] for f in steps[i]["fleets"] if f[1] == ai}
            my_fl_next = {f[0] for f in steps[i+1]["fleets"] if f[1] == ai}
            disappeared = my_fl_t - my_fl_next
            for fid in disappeared:
                f = next(f for f in steps[i]["fleets"] if f[0] == fid)
                if math.hypot(f[2] - 50, f[3] - 50) < 15:
                    sun_kills += 1

        # Overcommit: fleets launched with > needed+10% ships vs target garrison
        overcommit = 0
        total_launches = 0
        for step in steps:
            for move in (step["action"] or []):
                total_launches += 1
                # Find src planet and target
                src = next((p for p in step["planets"] if p[0] == move[0]), None)
                if src is None:
                    continue
                tgt_i = angle_target_planet(src, move[1], step["planets"])
                if tgt_i is None:
                    continue
                tgt = step["planets"][tgt_i]
                needed = tgt[5] + 1 if tgt[1] != ai else 0
                if needed and move[2] > needed * 1.1:
                    overcommit += 1

        # Clutch: WPA earned at steps where our P(win) < 0.4
        clutch = 0.0
        pwin_me = P[:, ai]
        for i in range(1, len(steps)):
            if pwin_me[i-1] < 0.4:
                clutch += pwin_me[i] - pwin_me[i-1]

        agent_rows.append({
            "episode_id": any_t["episode_id"],
            "agent_idx": ai,
            "team_name": t["team_name"],
            "winner": t["winner"],
            "n_players": n_players,
            "attack_ships_total": int(attack_ships),
            "conquests_held": conquests,
            "defense_rate": incoming_defended / incoming_total if incoming_total else 0.0,
            "incoming_attacks": incoming_total,
            "comet_grabs": comet_grabs,
            "sun_kills": sun_kills,
            "overcommit_count": overcommit,
            "launches_total": total_launches,
            "overcommit_rate": overcommit / total_launches if total_launches else 0.0,
            "clutch_wpa_from_behind": round(clutch, 4),
        })

    match_row = {
        "episode_id": any_t["episode_id"],
        "n_players": n_players,
        "n_steps": n_steps,
        "winner_agent": winner_idx,
        "gei": round(gei, 4),
        "comeback_magnitude": round(comeback_magnitude, 2),
        "num_lead_changes": num_lead_changes,
        "decisive_turn": decisive_turn if decisive_turn is not None else -1,
        "final_margin_rel": round(final_margin_rel, 4),
        "quality_tag": tag,
    }
    return match_row, agent_rows


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    traj_dir = pathlib.Path(args.traj_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = list(csv.DictReader((traj_dir / "index.csv").open()))
    # Load all trajectories, grouped by episode_id
    by_episode = defaultdict(list)
    all_trajs = []
    for row in idx:
        pkl = traj_dir / row["file"]
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            t = pickle.load(f)
        by_episode[int(t["episode_id"])].append(t)
        all_trajs.append(t)

    print(f"loaded {len(all_trajs)} trajectories across {len(by_episode)} episodes",
          flush=True)

    beta = fit_win_prob_model(all_trajs)
    if beta is None:
        print("ERROR: could not fit win-prob model (empty)", file=sys.stderr)
        return 1

    # Save model
    with (out_dir / "win_prob_model.pkl").open("wb") as f:
        pickle.dump({"beta": beta,
                     "features": ["ship_lead", "planet_lead", "prod_lead",
                                  "step_norm", "is_4p", "bias"]}, f)

    match_rows, agent_rows = [], []
    for ep_id, trajs in by_episode.items():
        try:
            m, a = analyze_episode(trajs, beta)
        except Exception as exc:
            print(f"  WARN episode {ep_id}: {exc}", flush=True)
            continue
        match_rows.append(m)
        agent_rows.extend(a)

    # Write
    if match_rows:
        with (out_dir / "analytics_match.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(match_rows[0].keys()))
            w.writeheader()
            w.writerows(match_rows)
    if agent_rows:
        with (out_dir / "analytics_agent.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agent_rows[0].keys()))
            w.writeheader()
            w.writerows(agent_rows)

    # Human-readable summary
    tags = defaultdict(int)
    for m in match_rows:
        tags[m["quality_tag"]] += 1
    print(f"match-level: {len(match_rows)} episodes  tags={dict(tags)}")
    print(f"agent-level: {len(agent_rows)} rows")
    print(f"wrote → {out_dir}/{{analytics_match.csv, analytics_agent.csv, win_prob_model.pkl}}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
