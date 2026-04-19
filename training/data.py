"""Dataset + collate for processed/<date>/*.npz files.

Each .npz holds ragged per-step arrays (see featurize.py). We flatten
across all episodes into a single "list of steps" and pad to the max
planet/fleet count in each batch.

Sampling weights by match-quality tag are supported: reads
processed/<date>/analytics_match.csv if present and up-weights
`epic`/`comeback` episodes, down-weights `blowout`.
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


SAMPLE_WEIGHTS = {
    "epic": 1.5,
    "comeback": 2.0,
    "close": 1.2,
    "standard": 1.0,
    "blowout": 0.2,  # keep a few, mostly drop
}


@dataclass
class Example:
    planets: np.ndarray        # [N, 14]
    planet_xy: np.ndarray      # [N, 2]
    fleets: np.ndarray         # [F, 9]
    globals_: np.ndarray       # [16]
    action_mask_owned: np.ndarray  # [N] bool
    # expert action labels, K rows each
    src: np.ndarray            # [K] int
    target: np.ndarray         # [K] int — index into planets
    bucket: np.ndarray         # [K] int


class BCDataset(Dataset):
    def __init__(self, proc_dir: pathlib.Path, keep_tags: Optional[set[str]] = None,
                 max_planets: int = 64, max_fleets: int = 64):
        self.proc_dir = pathlib.Path(proc_dir)
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        # Read feature index
        idx_csv = self.proc_dir / "index.csv"
        assert idx_csv.exists(), f"missing {idx_csv}"
        rows = list(csv.DictReader(idx_csv.open()))
        # Optional match-quality tags
        tag_by_ep: dict[int, str] = {}
        match_csv = self.proc_dir / "analytics_match.csv"
        if match_csv.exists():
            for m in csv.DictReader(match_csv.open()):
                tag_by_ep[int(m["episode_id"])] = m["quality_tag"]

        self.files: list[tuple[pathlib.Path, float, str]] = []
        for r in rows:
            path = self.proc_dir / r["file"]
            ep = int(r["episode_id"])
            tag = tag_by_ep.get(ep, "standard")
            if keep_tags is not None and tag not in keep_tags:
                continue
            weight = SAMPLE_WEIGHTS.get(tag, 1.0)
            self.files.append((path, weight, tag))

        # Flatten: index into per-episode, per-step
        self.episode_data: list[dict] = []
        self.step_index: list[tuple[int, int]] = []  # (episode_idx, step_idx)
        self.step_weight: list[float] = []
        for ei, (path, weight, _tag) in enumerate(self.files):
            z = np.load(path, allow_pickle=True)
            T = z["globals"].shape[0]
            self.episode_data.append({
                "planets": z["planets"],
                "planet_xy": z["planet_xy"],
                "fleets": z["fleets"],
                "globals": z["globals"],
                "action_mask_owned": z["action_mask_owned"],
                "src": z["src_planet_idx"],
                "target": z["target_planet_idx"],
                "bucket": z["ships_bucket"],
            })
            for ti in range(T):
                # Only include steps with at least one action (the label is
                # useful). We already filtered no-ops at featurize time by
                # default, but guard anyway.
                if len(self.episode_data[ei]["src"][ti]) == 0:
                    continue
                self.step_index.append((ei, ti))
                self.step_weight.append(weight)

    def __len__(self) -> int:
        return len(self.step_index)

    def __getitem__(self, i: int) -> Example:
        ei, ti = self.step_index[i]
        d = self.episode_data[ei]
        # Truncate ragged arrays to our padding limits
        planets = d["planets"][ti][: self.max_planets]
        planet_xy = d["planet_xy"][ti][: self.max_planets]
        fleets = d["fleets"][ti][: self.max_fleets]
        globals_ = d["globals"][ti]
        owned = d["action_mask_owned"][ti][: self.max_planets]
        src = d["src"][ti]
        target = d["target"][ti]
        bucket = d["bucket"][ti]
        # Drop moves whose src or target got truncated above max_planets
        valid = (src < len(planets)) & (target < len(planets))
        src = src[valid]
        target = target[valid]
        bucket = bucket[valid]
        return Example(
            planets=planets.astype(np.float32),
            planet_xy=planet_xy.astype(np.float32),
            fleets=fleets.astype(np.float32) if len(fleets) else
            np.zeros((0, 9), dtype=np.float32),
            globals_=globals_.astype(np.float32),
            action_mask_owned=owned.astype(bool),
            src=src.astype(np.int64),
            target=target.astype(np.int64),
            bucket=bucket.astype(np.int64),
        )


def collate(batch: list[Example]) -> dict:
    """Pad planets to max(P_batch), fleets to max(F_batch). Pack action
    labels into flat arrays with a batch_idx companion so the loss can
    index per-sample logits."""
    B = len(batch)
    P = max(ex.planets.shape[0] for ex in batch)
    F = max(ex.fleets.shape[0] for ex in batch) if any(
        ex.fleets.shape[0] for ex in batch) else 1

    planets = np.zeros((B, P, batch[0].planets.shape[1]), dtype=np.float32)
    planet_xy = np.zeros((B, P, 2), dtype=np.float32)
    planet_mask = np.zeros((B, P), dtype=bool)
    fleets = np.zeros((B, F, batch[0].fleets.shape[1] if batch[0].fleets.shape[0]
                       else 9), dtype=np.float32)
    fleet_mask = np.zeros((B, F), dtype=bool)
    globals_ = np.zeros((B, batch[0].globals_.shape[0]), dtype=np.float32)
    owned_mask = np.zeros((B, P), dtype=bool)

    flat_batch = []   # batch indices for each action label
    flat_src, flat_target, flat_bucket = [], [], []

    for i, ex in enumerate(batch):
        np_, _ = ex.planets.shape
        planets[i, :np_] = ex.planets
        planet_xy[i, :np_] = ex.planet_xy
        planet_mask[i, :np_] = True
        owned_mask[i, :np_] = ex.action_mask_owned
        nf = ex.fleets.shape[0]
        if nf > 0:
            fleets[i, :nf] = ex.fleets
            fleet_mask[i, :nf] = True
        globals_[i] = ex.globals_
        for s, t, b in zip(ex.src, ex.target, ex.bucket):
            flat_batch.append(i)
            flat_src.append(int(s))
            flat_target.append(int(t))
            flat_bucket.append(int(b))

    return {
        "planets": torch.from_numpy(planets),
        "planet_xy": torch.from_numpy(planet_xy),
        "planet_mask": torch.from_numpy(planet_mask),
        "fleets": torch.from_numpy(fleets),
        "fleet_mask": torch.from_numpy(fleet_mask),
        "globals": torch.from_numpy(globals_),
        "owned_mask": torch.from_numpy(owned_mask),
        "flat_batch": torch.tensor(flat_batch, dtype=torch.long),
        "flat_src": torch.tensor(flat_src, dtype=torch.long),
        "flat_target": torch.tensor(flat_target, dtype=torch.long),
        "flat_bucket": torch.tensor(flat_bucket, dtype=torch.long),
    }


def make_loader(proc_dir, batch_size=64, shuffle=True, num_workers=0,
                weighted=True, max_planets=64, max_fleets=64, **ds_kw):
    ds = BCDataset(proc_dir, max_planets=max_planets, max_fleets=max_fleets,
                   **ds_kw)
    if weighted and ds.step_weight and any(w != 1 for w in ds.step_weight):
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=ds.step_weight, num_samples=len(ds.step_index),
            replacement=True)
        shuffle = False  # exclusive with sampler
    else:
        sampler = None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler, collate_fn=collate,
                      num_workers=num_workers, drop_last=False)
