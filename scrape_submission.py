"""Track one of OUR submissions over time.

Usage:
  python3 scrape_submission.py --submission-id 51799179
  python3 scrape_submission.py --submission-id 51799179 --download-replays

Polls Kaggle's `EpisodeService/ListEpisodes` for a specific submissionId
via the same Playwright + intercepted-response pattern used by the
leaderboard scraper. Writes an append-only timeline so you can plot
rating / win-rate over time without re-scraping.

Output layout:
  tracking/<submission_id>/
      <YYYY-MM-DD>T<HH-MM-SS>.json    # one snapshot per run
      timeline.jsonl                  # flat timeline: one line per snapshot
      episodes/<episode_id>/          # only if --download-replays
          replay.json  episode.json

Each snapshot contains:
  {
    "snapshot_time": "...",
    "submission_id": <int>,
    "episodes": [...]   # raw ListEpisodes.episodes[] entries
  }
"""

import argparse
import datetime
import json
import pathlib
import re
import sys
import time

import requests
from playwright.sync_api import sync_playwright

LEADERBOARD_URL_TMPL = (
    "https://www.kaggle.com/competitions/orbit-wars/leaderboard?submissionId={sid}"
)
LIST_EPISODES_RE = re.compile(r"EpisodeService/ListEpisodes\b")
REPLAY_API = "https://www.kaggle.com/api/v1/competitions/episodes/{ep}/replay"


def load_kaggle_auth() -> tuple[str, str]:
    path = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    data = json.loads(path.read_text())
    return data["username"], data["key"]


def wait_for(page, probe: str, timeout_ms: int) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        try:
            if probe in page.content():
                return True
        except Exception:
            pass
        page.wait_for_timeout(1000)
    return False


def capture_list_episodes(submission_id: int, headless: bool = True) -> dict:
    """Navigate to the leaderboard URL with our submissionId query param —
    Kaggle auto-opens the episode dialog for that submission, firing a
    ListEpisodes call. We observe the response body."""
    pending: dict = {"body": None}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=["--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 1400, "height": 1200},
            user_agent=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
            bypass_csp=True,
        )
        page = context.new_page()

        def on_response(resp):
            if not LIST_EPISODES_RE.search(resp.url):
                return
            try:
                pending["body"] = resp.json()
            except Exception:
                pass

        page.on("response", on_response)

        url = LEADERBOARD_URL_TMPL.format(sid=submission_id)
        print(f"loading {url}", flush=True)
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        # Wait for hydration
        wait_for(page, "live_tv", timeout_ms=90000)
        page.wait_for_timeout(3000)

        # If the dialog didn't auto-open from the query param, click it
        # manually. The URL param approach works in the logged-out browser
        # most of the time — retry once via direct click if empty.
        deadline = time.time() + 15
        while time.time() < deadline and pending["body"] is None:
            page.wait_for_timeout(500)

        browser.close()

    return pending["body"] or {}


def fetch_replay(episode_id: int, auth: tuple[str, str]) -> dict | None:
    r = requests.get(REPLAY_API.format(ep=episode_id), auth=auth, timeout=120)
    if r.status_code != 200:
        return None
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-id", type=int, required=True)
    ap.add_argument("--download-replays", action="store_true",
                    help="Also pull full replay JSON for each new episode")
    ap.add_argument("--root", default=None,
                    help="Tracking root (default: <repo>/tracking/<sid>/)")
    args = ap.parse_args()

    sid = args.submission_id
    root = pathlib.Path(args.root) if args.root else (
        pathlib.Path(__file__).parent / "tracking" / str(sid))
    root.mkdir(parents=True, exist_ok=True)

    body = capture_list_episodes(sid)
    episodes = body.get("episodes") or body.get("Episodes") or []
    print(f"captured {len(episodes)} episodes for submission {sid}", flush=True)
    if not episodes:
        print("empty — probably rate-limited or the submission has no games yet",
              flush=True)
        return 1

    snapshot_time = datetime.datetime.now().isoformat(timespec="seconds")
    snapshot = {
        "snapshot_time": snapshot_time,
        "submission_id": sid,
        "episodes": episodes,
    }

    snapshot_path = root / f"{snapshot_time.replace(':', '-')}.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    # Append-only timeline for quick plotting
    timeline = root / "timeline.jsonl"
    with timeline.open("a", encoding="utf-8") as f:
        for e in episodes:
            # Extract our agent's (submissionId == sid) score update
            me = None
            for a in e.get("agents") or []:
                if a.get("submissionId") == sid:
                    me = a
                    break
            row = {
                "snapshot_time": snapshot_time,
                "episode_id": e.get("id"),
                "create_time": e.get("createTime"),
                "state": e.get("state"),
                "reward": (me or {}).get("reward"),
                "initial_score": (me or {}).get("initialScore"),
                "updated_score": (me or {}).get("updatedScore"),
                "opponents": [a.get("submissionId") for a in (e.get("agents") or [])
                              if a.get("submissionId") != sid],
            }
            f.write(json.dumps(row) + "\n")

    if args.download_replays:
        auth = load_kaggle_auth()
        ep_root = root / "episodes"
        ep_root.mkdir(exist_ok=True)
        for e in episodes:
            ep_id = e.get("id")
            if not ep_id:
                continue
            ep_dir = ep_root / str(ep_id)
            replay_path = ep_dir / "replay.json"
            if replay_path.exists() and replay_path.stat().st_size > 10_000:
                continue
            replay = fetch_replay(int(ep_id), auth)
            if replay is None:
                print(f"  ep {ep_id}: replay download failed", flush=True)
                continue
            ep_dir.mkdir(exist_ok=True)
            replay_path.write_text(json.dumps(replay), encoding="utf-8")
            (ep_dir / "episode.json").write_text(
                json.dumps(e, indent=2), encoding="utf-8")
            print(f"  ep {ep_id}: saved", flush=True)

    print(f"snapshot → {snapshot_path.relative_to(root.parent)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
