"""Fetch a specific Kaggle Orbit Wars episode replay by episodeId.

Usage:
  python fetch_episode_by_id.py --episode-id 75184271 --out replay_75184271.json

Uses Playwright to load the episode page (public, no auth needed) and
intercept the internal GetEpisodeReplay + GetEpisode API responses.

Reuses the same pattern as scrape_top5.py.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from playwright.sync_api import sync_playwright


EPISODE_URL_TEMPLATE = "https://www.kaggle.com/competitions/orbit-wars/episodes/{episode_id}"
REPLAY_API_FRAGMENT = "EpisodeService/GetEpisodeReplay"
META_API_FRAGMENT = "EpisodeService/GetEpisode"


def fetch(episode_id: int, out_dir: pathlib.Path, timeout_s: float = 45.0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    captured = {"replay": None, "meta": None}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()

        def on_response(resp):
            url = resp.url
            try:
                if REPLAY_API_FRAGMENT in url and captured["replay"] is None:
                    body = resp.body()
                    try:
                        captured["replay"] = json.loads(body)
                    except Exception:
                        captured["replay"] = {"_raw": body.decode(errors="replace")}
                elif META_API_FRAGMENT in url and captured["meta"] is None:
                    body = resp.body()
                    try:
                        captured["meta"] = json.loads(body)
                    except Exception:
                        captured["meta"] = {"_raw": body.decode(errors="replace")}
            except Exception as e:
                print(f"[warn] response capture failed for {url}: {e}", file=sys.stderr)

        page.on("response", on_response)

        url = EPISODE_URL_TEMPLATE.format(episode_id=episode_id)
        print(f"[fetch] navigating {url}", flush=True)
        page.goto(url, timeout=60_000, wait_until="networkidle")

        # Give APIs a moment to fire after hydration
        t0 = time.time()
        while (captured["replay"] is None or captured["meta"] is None) and (time.time() - t0) < timeout_s:
            time.sleep(1.0)

        html_path = out_dir / f"page_{episode_id}.html"
        html_path.write_text(page.content())
        screenshot_path = out_dir / f"page_{episode_id}.png"
        try:
            page.screenshot(path=str(screenshot_path))
        except Exception:
            pass

        browser.close()

    result = {
        "episode_id": episode_id,
        "replay_captured": captured["replay"] is not None,
        "meta_captured": captured["meta"] is not None,
    }
    if captured["replay"] is not None:
        replay_path = out_dir / f"replay_{episode_id}.json"
        replay_path.write_text(json.dumps(captured["replay"]))
        result["replay_path"] = str(replay_path)
        result["replay_size"] = replay_path.stat().st_size
    if captured["meta"] is not None:
        meta_path = out_dir / f"episode_{episode_id}.json"
        meta_path.write_text(json.dumps(captured["meta"], indent=2))
        result["meta_path"] = str(meta_path)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-id", type=int, required=True)
    ap.add_argument("--out-dir", default="simulation/adhoc")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    r = fetch(args.episode_id, out)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
