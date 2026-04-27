"""Build a Kaggle submission tar.gz for the k14 + lb1200 hybrid agent.

Sync helper modules from the live repo (so a stale copy in submission/
doesn't silently ship), then pack the bundle.

Usage:
  python submission/k14_lb1200_hybrid/build.py \
      [--ckpt training/checkpoints/<your_best>.pt]

Default ckpt: keep whatever model.pt is already in the bundle (so you
can drop a new ckpt in manually and re-run without --ckpt). To force a
specific source, pass --ckpt.

Output:
  submission/k14_lb1200_hybrid/submission.tar.gz

Submit with:
  kaggle competitions submit orbit-wars \
      -f submission/k14_lb1200_hybrid/submission.tar.gz \
      -m "<your message>"
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tarfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BUNDLE_DIR = pathlib.Path(__file__).resolve().parent
OUT_TAR = BUNDLE_DIR / "submission.tar.gz"

# (source path relative to REPO_ROOT) → (dest path relative to BUNDLE_DIR)
SYNC_MAP = {
    "featurize.py":                                 "featurize.py",
    "training/model.py":                            "training/model.py",
    "training/dual_stream_model.py":                "training/dual_stream_model.py",
    "training/lb1200_agent.py":                     "training/lb1200_agent.py",
    "training/physics_action_helper_k13.py":        "training/physics_action_helper_k13.py",
    "training/physics_picker_k13_ppo.py":           "training/physics_picker_k13_ppo.py",
    "training/hybrid_agent.py":                     "training/hybrid_agent.py",
}

# Files that must exist in the bundle to be valid
REQUIRED = ["main.py", "model.pt"] + list(SYNC_MAP.values())


def sync_helpers() -> None:
    """Refresh helper modules from the live repo."""
    for src_rel, dst_rel in SYNC_MAP.items():
        src = REPO_ROOT / src_rel
        dst = BUNDLE_DIR / dst_rel
        if not src.exists():
            raise FileNotFoundError(f"missing source: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  synced {src_rel} → {dst.relative_to(REPO_ROOT)}")
    init = BUNDLE_DIR / "training" / "__init__.py"
    init.touch(exist_ok=True)


def replace_ckpt(ckpt_src: str) -> None:
    src = pathlib.Path(ckpt_src)
    if not src.is_absolute():
        src = (REPO_ROOT / src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"ckpt not found: {src}")
    dst = BUNDLE_DIR / "model.pt"
    shutil.copy2(src, dst)
    print(f"  ckpt: {src.relative_to(REPO_ROOT)} → model.pt "
          f"({dst.stat().st_size / 1e6:.2f} MB)")


def verify_bundle() -> None:
    missing = [p for p in REQUIRED if not (BUNDLE_DIR / p).exists()]
    if missing:
        raise FileNotFoundError(
            "bundle incomplete; missing:\n  " + "\n  ".join(missing)
        )
    print(f"  bundle complete: {len(REQUIRED)} files present")


def pack() -> None:
    if OUT_TAR.exists():
        OUT_TAR.unlink()
    with tarfile.open(OUT_TAR, "w:gz") as tar:
        for fname in REQUIRED:
            src = BUNDLE_DIR / fname
            tar.add(src, arcname=fname)
    size_mb = OUT_TAR.stat().st_size / 1e6
    print(f"  packed → {OUT_TAR.relative_to(REPO_ROOT)} ({size_mb:.2f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="Replace model.pt with this checkpoint (relative "
                         "to repo root or absolute). Skip to keep current.")
    ap.add_argument("--skip-sync", action="store_true",
                    help="Don't re-copy helper modules; pack what's already "
                         "in the bundle dir.")
    args = ap.parse_args()

    print(f"bundle dir: {BUNDLE_DIR.relative_to(REPO_ROOT)}")
    if not args.skip_sync:
        print("syncing helpers from repo:")
        sync_helpers()
    if args.ckpt:
        print("replacing checkpoint:")
        replace_ckpt(args.ckpt)
    print("verifying bundle:")
    verify_bundle()
    print("packing tar.gz:")
    pack()

    print("\nSubmit with:")
    print(f"  kaggle competitions submit orbit-wars \\")
    print(f"      -f {OUT_TAR.relative_to(REPO_ROOT)} \\")
    print(f"      -m \"<your message>\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
