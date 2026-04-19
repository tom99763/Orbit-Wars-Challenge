"""Phase 1 Behaviour Cloning.

Train an OrbitAgent to imitate the top-10 winners' actions.

Usage:
  python training/bc.py --proc-dir processed/2026-04-19 \
                        --ckpt training/checkpoints/bc_v1.pt

Emits a Kaggle-grader-compatible .pt containing:
  - model state_dict
  - model init kwargs
  - training metadata (loss, date, n_examples)
"""

from __future__ import annotations

import argparse
import datetime
import math
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

# Make `training/` importable when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from training.data import make_loader   # noqa: E402
from training.model import OrbitAgent, sun_blocker_mask   # noqa: E402


def compute_bc_loss(batch, model, device):
    planets = batch["planets"].to(device)
    planet_xy = batch["planet_xy"].to(device)
    planet_mask = batch["planet_mask"].to(device)
    fleets = batch["fleets"].to(device)
    fleet_mask = batch["fleet_mask"].to(device)
    globals_ = batch["globals"].to(device)

    # NOTE: we trust the expert labels and skip the safety mask during
    # training. The expert sometimes takes shots our sun-blocker mask
    # would forbid (e.g. grazing trajectories that are actually safe
    # turn-by-turn). Applying the mask here would set -1e9 on the
    # labelled class for those samples and blow up the loss. The mask
    # is applied at inference (training/agent.py) where there's no label.
    tgt_logits, bkt_logits, value = model(
        planets, planet_mask, fleets, fleet_mask, globals_, target_mask=None,
    )
    # tgt_logits: [B,P,P+1]. pass=class 0; planet j = class j+1.
    fb = batch["flat_batch"].to(device)
    fs = batch["flat_src"].to(device)
    ft = batch["flat_target"].to(device)
    fk = batch["flat_bucket"].to(device)

    if fb.numel() == 0:
        return torch.tensor(0.0, device=device), 0.0, 0.0

    picked_tgt = tgt_logits[fb, fs]        # [K, P+1]
    picked_bkt = bkt_logits[fb, fs]        # [K, 4]
    # Labels: we pick target class = ft + 1 (0 is pass).
    tgt_labels = ft + 1
    loss_tgt = F.cross_entropy(picked_tgt, tgt_labels)
    loss_bkt = F.cross_entropy(picked_bkt, fk)
    loss = loss_tgt + 0.5 * loss_bkt

    with torch.no_grad():
        acc_tgt = (picked_tgt.argmax(-1) == tgt_labels).float().mean().item()
        acc_bkt = (picked_bkt.argmax(-1) == fk).float().mean().item()

    return loss, acc_tgt, acc_bkt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--weight-by-tag", action="store_true", default=True)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    loader = make_loader(args.proc_dir, batch_size=args.batch_size,
                         weighted=args.weight_by_tag)
    print(f"dataset size: {len(loader.dataset)} steps from "
          f"{len(loader.dataset.files)} trajectories", flush=True)

    kwargs = dict(planet_dim=14, fleet_dim=9, global_dim=16,
                  d_model=args.d_model, n_heads=4, n_layers=args.n_layers,
                  n_buckets=4)
    model = OrbitAgent(**kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f} M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(loader)
    )

    best_loss = float("inf")
    log_rows = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        for batch in loader:
            loss, at, ab = compute_bc_loss(batch, model, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            epoch_loss.append(loss.item())
            if step % args.log_every == 0:
                print(f"[ep {epoch:02d} step {step:05d}] loss={loss.item():.3f} "
                      f"tgt_acc={at:.3f} bkt_acc={ab:.3f} lr={sched.get_last_lr()[0]:.1e}",
                      flush=True)
            step += 1
        mean_loss = sum(epoch_loss) / len(epoch_loss)
        log_rows.append({"epoch": epoch, "loss": mean_loss})
        print(f"[ep {epoch:02d}] mean_loss={mean_loss:.3f}  "
              f"elapsed={time.time()-t0:.1f}s", flush=True)

        if mean_loss < best_loss:
            best_loss = mean_loss
            ckpt_path = pathlib.Path(args.ckpt)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "kwargs": kwargs,
                "best_loss": best_loss,
                "n_examples": len(loader.dataset),
                "epoch": epoch,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "args": vars(args),
                "log": log_rows,
            }, ckpt_path)
            print(f"  → saved {ckpt_path} (best={best_loss:.4f})", flush=True)

    print(f"done  best_loss={best_loss:.4f}  total={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
