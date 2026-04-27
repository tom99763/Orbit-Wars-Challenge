"""BC auxiliary loss for k14 self-play training.

Lets the PPO trainer in physics_picker_k14_vec.py mix in a behavioural-
cloning signal from scraped top-N expert data. The goal is regularization,
not pure imitation: a small BC term widens the policy's training
distribution so it doesn't collapse into "beats only opponents that look
like itself".

Public API:
    BCLoader(data_dir, batch_size, device, ...)
        .sample()  → batch dict with planets/fleets/globals/spatial/...
                     plus mode_label, frac_label

    compute_bc_loss(net, batch) → (total_loss, info_dict)
        info_dict has scalar metrics for logging:
            mode_ce, mode_acc, mode_n,
            frac_ce, frac_acc, frac_n
        — n keys are how many examples the loss covers, useful when
        winners-only filtering or label-yield drops batch coverage.

Padding: variable-length planets/fleets per step are padded to the batch
maximum at sample time. Padded slots get planet_mask = False and
mode_label = -1 (ignored by CE via ignore_index).
"""
from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def _pad_2d(arrs: list[np.ndarray], max_n: int, pad_value: float = 0.0) -> np.ndarray:
    """Stack list of (N_i, D) → (B, max_n, D), zero-pad on N."""
    B = len(arrs)
    if B == 0:
        return np.zeros((0, max_n, 0), dtype=np.float32)
    D = arrs[0].shape[1] if arrs[0].ndim >= 2 else 1
    out = np.full((B, max_n, D), pad_value, dtype=np.float32)
    for i, a in enumerate(arrs):
        n = a.shape[0]
        out[i, :n] = a
    return out


def _pad_mask(arrs: list[np.ndarray], max_n: int) -> np.ndarray:
    """For each (N_i,) input, build a (B, max_n) bool mask True for valid slots."""
    B = len(arrs)
    out = np.zeros((B, max_n), dtype=bool)
    for i, a in enumerate(arrs):
        out[i, : a.shape[0]] = True
    return out


def _pad_int_label(arrs: list[np.ndarray], max_n: int, pad_value: int = -1) -> np.ndarray:
    """Pad (N_i,) integer label arrays to (B, max_n), padding with -1."""
    B = len(arrs)
    out = np.full((B, max_n), pad_value, dtype=np.int64)
    for i, a in enumerate(arrs):
        out[i, : a.shape[0]] = a.astype(np.int64)
    return out


@dataclass
class _ShardIndex:
    """Index into one BC dataset shard. Shards are loaded lazily into a
    cache so we don't pay npz-load cost per sample."""
    path: pathlib.Path
    n_steps: int
    is_winner: bool


class BCLoader:
    """Random-access sampler over a directory of build_bc_dataset shards.

    All shards are scanned at construction (fast, only reads metadata).
    Step examples are sampled uniformly across all (shard, step) pairs.

    Args:
        data_dir:     directory containing bc_*.npz shards
        batch_size:   B, number of (state, label) examples per sample
        device:       torch device
        winners_only: keep only shards with is_winner=True
        shard_cache:  max number of shards held in memory at once
        seed:         RNG seed for reproducibility
    """

    def __init__(
        self,
        data_dir: str | pathlib.Path,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
        winners_only: bool = False,
        shard_cache: int = 8,
        seed: int | None = None,
    ):
        self.data_dir = pathlib.Path(data_dir)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shards: list[_ShardIndex] = []
        self._cache: dict[str, dict] = {}
        self._cache_order: list[str] = []
        self._cache_max = shard_cache
        self.rng = random.Random(seed)

        for p in sorted(self.data_dir.glob("bc_*.npz")):
            try:
                with np.load(p, allow_pickle=True) as z:
                    is_winner = bool(z["is_winner"])
                    n_steps = int(z["n_steps"])
            except Exception:
                continue
            if winners_only and not is_winner:
                continue
            if n_steps <= 0:
                continue
            self.shards.append(_ShardIndex(path=p, n_steps=n_steps, is_winner=is_winner))

        if not self.shards:
            raise RuntimeError(
                f"No usable BC shards found under {self.data_dir} "
                f"(winners_only={winners_only})."
            )

        # Flat (shard_idx, step_idx) index for uniform sampling
        self._flat: list[tuple[int, int]] = []
        for si, sh in enumerate(self.shards):
            for t in range(sh.n_steps):
                self._flat.append((si, t))

        print(f"[BCLoader] {len(self.shards)} shards, "
              f"{len(self._flat)} steps total "
              f"(winners_only={winners_only})", flush=True)

    def _get_shard(self, idx: int) -> dict:
        sh = self.shards[idx]
        key = str(sh.path)
        if key in self._cache:
            return self._cache[key]
        with np.load(sh.path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        self._cache[key] = data
        self._cache_order.append(key)
        while len(self._cache_order) > self._cache_max:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        return data

    def __len__(self) -> int:
        return len(self._flat)

    def sample(self) -> dict:
        """Sample `batch_size` random examples and return a padded batch
        dict ready for net.forward + compute_bc_loss."""
        picks = self.rng.sample(
            self._flat, k=min(self.batch_size, len(self._flat))
        )
        planets, fleets, globals_, spatial = [], [], [], []
        mode_lbls, frac_lbls = [], []

        for shard_idx, t in picks:
            shard = self._get_shard(shard_idx)
            planets.append(np.asarray(shard["planets"][t], dtype=np.float32))
            fleets.append(np.asarray(shard["fleets"][t], dtype=np.float32))
            globals_.append(np.asarray(shard["globals"][t], dtype=np.float32))
            spatial.append(np.asarray(shard["spatial"][t], dtype=np.float32))
            mode_lbls.append(np.asarray(shard["mode_label"][t]))
            frac_lbls.append(np.asarray(shard["frac_label"][t]))

        # Determine batch padding sizes
        max_p = max((p.shape[0] for p in planets), default=1)
        max_f = max((f.shape[0] for f in fleets), default=1)
        max_p = max(max_p, 1)
        max_f = max(max_f, 1)

        # Some steps have zero fleets — featurize emits (0, FLEET_DIM) — pad to (1, D)
        normed_fleets = []
        for f in fleets:
            if f.ndim < 2 or f.shape[0] == 0:
                from featurize import FLEET_DIM
                normed_fleets.append(np.zeros((0, FLEET_DIM), dtype=np.float32))
            else:
                normed_fleets.append(f)
        fleets = normed_fleets

        planets_t = torch.from_numpy(_pad_2d(planets, max_p)).to(self.device)
        planet_mask_t = torch.from_numpy(_pad_mask(planets, max_p)).to(self.device)
        fleets_t = torch.from_numpy(_pad_2d(fleets, max_f)).to(self.device)
        fleet_mask_t = torch.from_numpy(_pad_mask(fleets, max_f)).to(self.device)
        globals_t = torch.from_numpy(np.stack(globals_, axis=0)).to(self.device)
        spatial_t = torch.from_numpy(np.stack(spatial, axis=0)).to(self.device)
        mode_t = torch.from_numpy(_pad_int_label(mode_lbls, max_p)).to(self.device)
        frac_t = torch.from_numpy(_pad_int_label(frac_lbls, max_p)).to(self.device)

        return {
            "planets": planets_t,
            "planet_mask": planet_mask_t,
            "fleets": fleets_t,
            "fleet_mask": fleet_mask_t,
            "globals": globals_t,
            "spatial": spatial_t,
            "mode_label": mode_t,
            "frac_label": frac_t,
        }


def compute_bc_loss(net, batch: dict) -> tuple[torch.Tensor, dict]:
    """Forward `batch` through `net` and return (total_loss, info_dict).

    Mode CE uses ignore_index=-1 to skip planets that are
    not-owned-by-expert or whose action couldn't be recovered.
    Frac CE only applies on planets where mode_label > 0 (mode 0 = pass
    has no fraction); these planets feed mode-conditioned frac_logits_for.
    """
    planets = batch["planets"]
    planet_mask = batch["planet_mask"]
    fleets = batch["fleets"]
    fleet_mask = batch["fleet_mask"]
    globals_ = batch["globals"]
    spatial = batch["spatial"]
    mode_label = batch["mode_label"]   # (B, P), -1 = ignore
    frac_label = batch["frac_label"]   # (B, P), -1 = ignore

    fused_planet_tokens, mode_logits, _value = net(
        planets, planet_mask, fleets, fleet_mask, globals_, spatial,
    )
    B, P, n_modes = mode_logits.shape

    # Mode CE
    mode_loss = F.cross_entropy(
        mode_logits.reshape(-1, n_modes),
        mode_label.reshape(-1),
        ignore_index=-1,
        reduction="mean",
    )
    with torch.no_grad():
        mode_valid = mode_label.reshape(-1) >= 0
        mode_n = int(mode_valid.sum().item())
        if mode_n > 0:
            mode_acc = (
                mode_logits.reshape(-1, n_modes).argmax(-1)[mode_valid]
                == mode_label.reshape(-1)[mode_valid]
            ).float().mean().item()
        else:
            mode_acc = 0.0

    # Frac CE — only on labelled active planets
    active = (mode_label > 0) & (frac_label >= 0)
    n_active = int(active.sum().item())
    if n_active == 0:
        frac_loss = torch.zeros((), device=planets.device)
        frac_acc = 0.0
    else:
        active_b, active_p = active.nonzero(as_tuple=True)
        sel_tokens = fused_planet_tokens[active_b, active_p]   # (N, d)
        sel_modes = mode_label[active_b, active_p]             # (N,)
        sel_targets = frac_label[active_b, active_p]           # (N,)
        frac_logits = net.frac_logits_for(sel_tokens, sel_modes)  # (N, n_fracs)
        frac_loss = F.cross_entropy(
            frac_logits, sel_targets.long(), reduction="mean"
        )
        with torch.no_grad():
            frac_acc = (frac_logits.argmax(-1) == sel_targets).float().mean().item()

    total = mode_loss + frac_loss
    info = {
        "mode_ce": mode_loss.detach().item(),
        "mode_acc": mode_acc,
        "mode_n": mode_n,
        "frac_ce": frac_loss.detach().item() if n_active else 0.0,
        "frac_acc": frac_acc,
        "frac_n": n_active,
    }
    return total, info
