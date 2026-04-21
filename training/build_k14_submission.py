"""Build orbit_wars_k14.py: k14 DualStreamK13Agent (mode×frac), pure numpy, no torch."""
import pathlib, base64

SRC = pathlib.Path("notebooks/orbit_wars_v3_numpy.py")
NPZ = pathlib.Path("training/physics_picker_k14_weights.npz")
DST = pathlib.Path("notebooks/orbit_wars_k14.py")

npz_b64 = base64.b64encode(NPZ.read_bytes()).decode()
print(f"NPZ base64 length: {len(npz_b64):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. _forward replacement inside _init_neural_model
#    v3 uses cand_head; k14 uses mode_head + frac_head (mode-conditional)
# ─────────────────────────────────────────────────────────────────────────────

OLD_FORWARD_TAIL = """            fused_p = planet_tokens + fused_g[:, None, :]
            logits  = _mm(fused_p, W['cand_head.weight'], W['cand_head.bias'])
            value   = _mm(
                _gelu(_mm(fused_g, W['value_head.0.weight'], W['value_head.0.bias'])),
                W['value_head.2.weight'], W['value_head.2.bias']).squeeze(-1)
            return logits, value"""

NEW_FORWARD_TAIL = """            fused_p = planet_tokens + fused_g[:, None, :]
            mode_logits = _mm(fused_p, W['mode_head.weight'], W['mode_head.bias'])
            frac_logits_all = _np.stack([
                _mm(fused_p + W['mode_embed.weight'][m][_np.newaxis, _np.newaxis, :],
                    W['frac_head.weight'], W['frac_head.bias'])
                for m in range(5)
            ], axis=2)
            return mode_logits, frac_logits_all"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. k14 helper functions inserted before _SESSION = None
# ─────────────────────────────────────────────────────────────────────────────

K14_HELPERS = '''
# k14 mode × fraction helpers (inline from physics_action_helper_k13)
_K14_FRACTIONS = [0.05, 0.15, 0.30, 0.50, 0.65, 0.80, 0.95, 1.00]
_K14_MIN_SHIPS = 2
_K14_MIN_KEEP  = 1

def _k14_mode_mask(src, world, my_player):
    """Return [5] bool list: True = mode valid for this source planet."""
    planets = world.planets
    def reach(p):
        if p.id == src.id: return False
        if segment_hits_sun(src.x, src.y, p.x, p.y): return False
        return math.hypot(p.x - src.x, p.y - src.y) <= MAX_DISTANCE
    have_neutral = any(p.owner == -1              and reach(p) for p in planets)
    have_enemy   = any(p.owner != my_player and p.owner != -1 and reach(p) for p in planets)
    have_friend  = any(p.owner == my_player and p.id != src.id and reach(p) for p in planets)
    any_enemy    = any(p.owner != my_player and p.owner != -1  for p in planets)
    return [True, have_neutral, have_enemy, have_friend,
            any_enemy and (have_neutral or have_enemy)]


def _k14_find_target(src, world, my_player, mode):
    """Pick best target planet for given mode; None if nothing reachable."""
    planets = world.planets
    def reach(p):
        if p.id == src.id: return False
        if segment_hits_sun(src.x, src.y, p.x, p.y): return False
        return math.hypot(p.x - src.x, p.y - src.y) <= MAX_DISTANCE
    if mode == 1:
        cands = [p for p in planets if p.owner == -1 and reach(p)]
    elif mode == 2:
        cands = [p for p in planets if p.owner != my_player and p.owner != -1 and reach(p)]
    elif mode == 3:
        cands = [p for p in planets if p.owner == my_player and p.id != src.id and reach(p)]
    elif mode == 4:
        enemies = [p for p in planets if p.owner != my_player and p.owner != -1]
        cands = []
        for p in planets:
            if p.owner == my_player or not reach(p):
                continue
            for e in enemies:
                me_to_p = math.hypot(p.x - src.x, p.y - src.y)
                p_to_e  = math.hypot(e.x - p.x,   e.y - p.y)
                me_to_e = math.hypot(e.x - src.x, e.y - src.y) + 1e-6
                if (me_to_p + p_to_e) - me_to_e < 15.0:
                    cands.append(p)
                    break
    else:
        return None
    if not cands:
        return None
    if mode == 3:
        cands.sort(key=lambda p: (p.ships, math.hypot(p.x - src.x, p.y - src.y)))
    else:
        cands.sort(key=lambda p: math.hypot(p.x - src.x, p.y - src.y))
    return cands[0]


def _k14_materialize(picks, world, my_player):
    """Given [(pid, mode_idx, frac_idx), ...], return env action list."""
    planet_by_id = {p.id: p for p in world.planets}
    initial_by_id = getattr(world, 'initial_by_id', {})
    ang_vel  = getattr(world, 'ang_vel', 0.0)
    comets   = getattr(world, 'comets', []) or []
    comet_ids = getattr(world, 'comet_ids', set())
    actions = []
    for pid, mode, frac_idx in picks:
        src = planet_by_id.get(int(pid))
        if src is None or src.owner != my_player or mode == 0:
            continue
        tgt = _k14_find_target(src, world, my_player, mode)
        if tgt is None:
            continue
        frac  = _K14_FRACTIONS[int(frac_idx)]
        ships = max(1, int(src.ships * frac))
        ships = min(ships, src.ships - _K14_MIN_KEEP)
        if ships < _K14_MIN_SHIPS:
            continue
        aim = aim_with_prediction(src, tgt, ships, initial_by_id, ang_vel,
                                  comets=comets, comet_ids=comet_ids)
        if aim is None:
            continue
        angle = aim[0] if isinstance(aim, tuple) else aim
        actions.append([int(src.id), float(angle), int(ships)])
    return actions

'''

# ─────────────────────────────────────────────────────────────────────────────
# 3. _agent_impl replacement
#    Old: candidate-based (cand_head logits → pick from K_PER_SOURCE candidates)
#    New: mode × frac (mode_head + frac_head → _k14_materialize)
# ─────────────────────────────────────────────────────────────────────────────

OLD_AGENT_IMPL_CORE = """    logits_np_b, _v = _NP_FORWARD(
        pl[np.newaxis],
        pmask[np.newaxis],
        fl[np.newaxis],
        fmask[np.newaxis],
        feat["globals"][np.newaxis],
        spatial[np.newaxis],
    )
    logits_np = logits_np_b[0]

    # Only process top-N owned planets by logit score to stay within actTimeout=1s
    _MAX_PLANETS = 4
    owned_indices = [
        (i, pid) for i, pid in enumerate(feat.get("planet_ids", []))
        if any(p.id == int(pid) and p.owner == my_player for p in world.planets)
    ]
    # rank by max logit (model confidence), take top N
    owned_indices.sort(key=lambda x: logits_np[x[0]].max(), reverse=True)
    owned_indices = owned_indices[:_MAX_PLANETS]

    comets = getattr(world, "comets", []) or []
    picks = []
    for i, pid in owned_indices:
        src = next((p for p in world.planets if p.id == int(pid)), None)
        if src is None:
            continue
        row = logits_np[i].copy()
        cands = generate_per_source_candidates(src, world, my_player, comets)
        # mask out None (pass) candidates — guarantee a real action
        for k, c in enumerate(cands):
            if c is None:
                row[k] = -1e9
        if row.max() < -1e8:
            continue  # all candidates are None, skip this planet
        probs = np.exp(row - row.max())
        probs = probs / probs.sum()
        ci = int(np.random.choice(K_PER_SOURCE, p=probs))
        picks.append((int(pid), ci))

    action_list = materialize_joint_action(picks, world, my_player)"""

NEW_AGENT_IMPL_CORE = """    mode_logits_b, frac_logits_all_b = _NP_FORWARD(
        pl[np.newaxis],
        pmask[np.newaxis],
        fl[np.newaxis],
        fmask[np.newaxis],
        feat["globals"][np.newaxis],
        spatial[np.newaxis],
    )
    mode_logits_np     = mode_logits_b[0]       # (P, 5)
    frac_logits_all_np = frac_logits_all_b[0]   # (P, 5, 8)

    picks = []  # (pid, mode_idx, frac_idx)
    planet_by_id = {p.id: p for p in world.planets}
    for i, pid in enumerate(feat.get("planet_ids", [])):
        src = planet_by_id.get(int(pid))
        if src is None or src.owner != my_player:
            continue
        mm = _k14_mode_mask(src, world, my_player)
        ml = mode_logits_np[i].copy()
        for mi in range(5):
            if not mm[mi]:
                ml[mi] = -1e9
        m_p = np.exp(ml - ml.max()); m_p = m_p / m_p.sum()
        mode_idx = int(np.random.choice(5, p=m_p))
        if mode_idx == 0:
            continue
        fl = frac_logits_all_np[i, mode_idx].copy()
        f_p = np.exp(fl - fl.max()); f_p = f_p / f_p.sum()
        frac_idx = int(np.random.choice(8, p=f_p))
        picks.append((int(pid), mode_idx, frac_idx))

    action_list = _k14_materialize(picks, world, my_player)"""

# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────
lines = SRC.read_bytes().decode("utf-8")
orig_len = len(lines)

# 1. Replace _WEIGHT_NPZ_B64
b64_start = lines.index('\n_WEIGHT_NPZ_B64 = (\n')
b64_end   = lines.index('\n)\n', b64_start) + 3
old_b64_block = lines[b64_start:b64_end]
new_b64_block = f'\n_WEIGHT_NPZ_B64 = (\n    "{npz_b64}"\n)\n'
lines = lines[:b64_start] + new_b64_block + lines[b64_end:]
print(f"[1] Replaced _WEIGHT_NPZ_B64 ({len(old_b64_block):,} → {len(new_b64_block):,})")

# 2. Replace _forward tail (cand_head → mode/frac heads)
assert OLD_FORWARD_TAIL in lines, "ERROR: old _forward tail not found"
lines = lines.replace(OLD_FORWARD_TAIL, NEW_FORWARD_TAIL, 1)
print("[2] Replaced _forward tail (cand_head → mode_head + frac_head)")

# 3. Insert k14 helpers before _SESSION = None
HELPERS_ANCHOR = '\n_SESSION = None\n'
assert HELPERS_ANCHOR in lines, "ERROR: _SESSION = None anchor not found"
lines = lines.replace(HELPERS_ANCHOR, '\n' + K14_HELPERS + '_SESSION = None\n', 1)
print("[3] Inserted k14 helpers (_k14_mode_mask, _k14_find_target, _k14_materialize)")

# 4. Replace _agent_impl inference + action selection core
assert OLD_AGENT_IMPL_CORE in lines, "ERROR: old _agent_impl core not found"
lines = lines.replace(OLD_AGENT_IMPL_CORE, NEW_AGENT_IMPL_CORE, 1)
print("[4] Replaced _agent_impl inference core (candidate → mode×frac)")

# 5. Write
DST.write_bytes(lines.encode("utf-8"))
size_mb = DST.stat().st_size / 1e6
print(f"\n[done] {DST}  ({size_mb:.2f} MB)")
