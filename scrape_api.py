"""API-based Orbit Wars replay scraper.

Replaces the Playwright-only scrape_top5.py with a hybrid approach:

  Step 1 (discovery): Playwright opens the leaderboard and clicks each top-N
         team's live_tv icon. Each click triggers Kaggle's internal
         `EpisodeService/ListEpisodes` call; we observe the response and
         extract the most-recent episode id per team. We do NOT wait for
         the full replay JSON to load — only the small ListEpisodes payload.
         This bottleneck stays Playwright-based because the endpoint is
         gated by a reCAPTCHA for non-browser clients.

  Step 2 (download): pure `requests` + Basic Auth against
         https://www.kaggle.com/api/v1/competitions/episodes/{id}/replay
         (the public endpoint used by download_replay.py). No browser needed.

  Step 3 (offline player): optional — preserve the pre-existing behaviour
         of emitting `play_<id>.html` + a shared `player_assets/` bundle so
         replays can be watched locally.

Output layout is unchanged from scrape_top5.py:

  simulation/<YYYY-MM-DD>/<HH-MM>/
      leaderboard.html / .png        (optional — --no-screenshot skips)
      player_assets/                 (one copy shared across teams)
      rank0X_<team>/
          replay_<ep>.json           (full per-step replay, from API)
          episode_<ep>.json          (condensed metadata extracted from replay)
          play_<ep>.html             (offline wrapper)

Auth:
  Reads ~/.kaggle/kaggle.json for (username, key). The replay download
  endpoint returns 401 with a stale key — rotate at
  https://www.kaggle.com/settings/account → API → Create New Token.
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import sys
import time
import urllib.request
from typing import Optional

import requests
from playwright.sync_api import sync_playwright

LEADERBOARD_URL = "https://www.kaggle.com/competitions/orbit-wars/leaderboard"
VISUALIZER_INDEX = "https://www.kaggleusercontent.com/episode-visualizers/orbit_wars/default/index.html"
REPLAY_API = "https://www.kaggle.com/api/v1/competitions/episodes/{ep}/replay"
LIST_EPISODES_RE = re.compile(r"EpisodeService/ListEpisodes\b")

TOP_N = 10
EPISODES_PER_TEAM = 5  # how many recent matchups to pull per team


def slugify(name: str) -> str:
    """ASCII-preferring slug with a URL-safe fallback for all-unicode names.
    e.g. 'Shun_PI' → 'Shun_PI', 'Ha Quang Minh' → 'Ha_Quang_Minh',
    '寿!' → 'team_9af3' (4 hex chars from the sha1, stable per team name).
    """
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    if s:
        return s
    import hashlib
    return f"team_{hashlib.sha1(name.encode('utf-8')).hexdigest()[:4]}"


def load_kaggle_auth() -> tuple[str, str]:
    path = pathlib.Path(os.environ.get("KAGGLE_CONFIG_DIR", pathlib.Path.home() / ".kaggle")) / "kaggle.json"
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


def get_top_n_teams(page, n: int) -> list[str]:
    """Top-N team names in leaderboard order, from the hydrated DOM."""
    return page.evaluate(
        """(n) => {
            const uls = Array.from(document.querySelectorAll('ul[role="list"]'));
            const ul = uls.find(u =>
                u.querySelector('.google-symbols') &&
                Array.from(u.querySelectorAll('li')).some(li => /^\\s*\\d+/.test(li.innerText || '')));
            if (!ul) return [];
            const out = [];
            for (const li of ul.querySelectorAll(':scope > li')) {
                const lines = (li.innerText || '').split('\\n').map(s => s.trim()).filter(Boolean);
                if (lines.length < 2 || !/^\\d+$/.test(lines[0])) continue;
                out.push(lines[1]);
                if (out.length >= n) break;
            }
            return out;
        }""",
        n,
    )


def click_live_tv(page, team_name: str) -> str:
    return page.evaluate(
        """(teamName) => {
            const uls = Array.from(document.querySelectorAll('ul[role="list"]'));
            const ul = uls.find(u =>
                u.querySelector('.google-symbols') &&
                Array.from(u.querySelectorAll('li')).some(li => /^\\s*\\d+/.test(li.innerText || '')));
            if (!ul) return 'no_ul';
            for (const li of ul.querySelectorAll(':scope > li')) {
                if (!li.innerText.includes(teamName)) continue;
                for (const el of li.querySelectorAll('.google-symbols')) {
                    if ((el.innerText || '').trim() === 'live_tv') {
                        el.scrollIntoView({block: 'center'});
                        el.click();
                        return 'ok';
                    }
                }
                return 'no_icon';
            }
            return 'no_row';
        }""",
        team_name,
    )


def close_dialog(page) -> None:
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(800)
    except Exception:
        pass


def discover_top_episodes(top_n: int, episodes_per_team: int,
                          headless: bool = True) -> list[dict]:
    """Return [{'rank', 'team', 'episode_id', 'create_time', 'agents', ...}]
    for the top-N teams, up to `episodes_per_team` recent matchups each.

    Uses Playwright to click each team's live_tv icon and observes the
    ListEpisodes response. The Kaggle UI also reflects the selected episode
    in the URL as `?submissionId=X&episodeId=Y` — we don't rely on it
    because the raw JSON gives the same ids without another click.
    """
    results: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=["--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 1600, "height": 1400},
            user_agent=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
            bypass_csp=True,
        )
        page = context.new_page()

        pending: dict = {"body": None}

        def on_response(resp):
            if not LIST_EPISODES_RE.search(resp.url):
                return
            try:
                pending["body"] = resp.json()
            except Exception:
                pass

        page.on("response", on_response)

        print(f"loading {LEADERBOARD_URL}", flush=True)
        page.goto(LEADERBOARD_URL, wait_until="domcontentloaded", timeout=60000)
        wait_for(page, "Shun_PI", timeout_ms=90000)
        page.wait_for_timeout(3000)

        top = get_top_n_teams(page, top_n)
        print(f"top {top_n}: {top}", flush=True)

        for rank, name in enumerate(top, start=1):
            pending["body"] = None
            res = click_live_tv(page, name)
            print(f"[rank {rank}] {name}  live_tv={res}", flush=True)
            if res != "ok":
                continue

            try:
                page.wait_for_selector(".MuiModal-root", timeout=15000)
            except Exception:
                print("  WARN: modal not seen", flush=True)

            deadline = time.time() + 15
            while time.time() < deadline and pending["body"] is None:
                page.wait_for_timeout(400)

            body = pending["body"] or {}
            episodes = body.get("episodes") or body.get("Episodes") or []
            if not episodes:
                print("  WARN: no episodes in ListEpisodes response", flush=True)
                close_dialog(page)
                continue

            picked = episodes[:episodes_per_team]
            print(f"  {len(picked)} episode(s): {[e.get('id') for e in picked]}",
                  flush=True)
            for e in picked:
                results.append({
                    "rank": rank,
                    "team": name,
                    "episode_id": int(e.get("id") or e.get("Id")),
                    "create_time": e.get("createTime"),
                    "end_time": e.get("endTime"),
                    "state": e.get("state"),
                    "agents": e.get("agents"),
                })

            close_dialog(page)

        browser.close()

    return results


def download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())


def fetch_visualizer_bundle(assets_dir: pathlib.Path) -> pathlib.Path:
    assets_dir.mkdir(parents=True, exist_ok=True)
    index_path = assets_dir / "index.html"
    download(VISUALIZER_INDEX, index_path)
    html = index_path.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r'(?:src|href)="(\.\/[^"#?]+|assets\/[^"#?]+)"', html):
        rel = m.group(1).lstrip("./")
        url = f"https://www.kaggleusercontent.com/episode-visualizers/orbit_wars/default/{rel}"
        try:
            download(url, assets_dir / rel)
        except Exception as e:
            print(f"  (warn) could not fetch {url}: {e}", flush=True)
    return index_path


def make_offline_player(episode_dir: pathlib.Path, episode_id: int,
                        replay_path: pathlib.Path, assets_rel: str) -> pathlib.Path:
    out = episode_dir / "play.html"
    out.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Episode {episode_id}</title>
<style>html,body{{margin:0;height:100%;background:#000;color:#eee;font-family:sans-serif}}
iframe{{width:100%;height:100vh;border:0}}</style></head><body>
<iframe id="player" src="{assets_rel}/index.html"></iframe>
<script>
  fetch({json.dumps(replay_path.name)})
    .then(r => r.json())
    .then(env => {{
      const f = document.getElementById('player');
      const send = () => f.contentWindow.postMessage({{ type: 'update', environment: env }}, '*');
      f.addEventListener('load', send);
      if (f.contentDocument && f.contentDocument.readyState === 'complete') send();
      setTimeout(send, 1000);
      setTimeout(send, 3000);
    }});
</script></body></html>
""",
        encoding="utf-8",
    )
    return out


def fetch_replay(episode_id: int, auth: tuple[str, str]) -> Optional[dict]:
    r = requests.get(REPLAY_API.format(ep=episode_id), auth=auth, timeout=120)
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.content[:200]!r}", flush=True)
        return None
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                    help="Output dir. Default: simulation/<YYYY-MM-DD>/<HH-MM>/")
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--episodes-per-team", type=int, default=EPISODES_PER_TEAM,
                    help="How many recent matchups to download per team")
    ap.add_argument("--no-player", action="store_true",
                    help="Skip downloading visualizer bundle + play_<id>.html wrappers")
    args = ap.parse_args()

    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir)
    else:
        # Per-day rollup. Each episode lands under its team folder:
        #   simulation/<YYYY-MM-DD>/<team_slug>/<episode_id>/replay.json
        # Hourly cron runs accumulate into the same day folder, deduped by
        # episode_id, so you get up to (top_n × episodes_per_team × 24) unique
        # games per day in the worst case — usually far fewer because the
        # same recent matchups are returned within an hour.
        now = datetime.datetime.now()
        out_dir = (pathlib.Path(__file__).parent / "simulation"
                   / now.date().isoformat())
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_dir}  top_n: {args.top_n}", flush=True)

    auth = load_kaggle_auth()

    if not args.no_player:
        # Shared visualizer bundle lives at the day root — one copy for the
        # whole day. `play.html` inside each episode folder references
        # ../../player_assets/ relative to itself.
        assets_dir = out_dir / "player_assets"
        if not (assets_dir / "index.html").exists():
            print("fetching visualizer bundle …", flush=True)
            fetch_visualizer_bundle(assets_dir)

    # Step 1: discover episode IDs
    entries = discover_top_episodes(args.top_n, args.episodes_per_team)
    if not entries:
        print("no episodes discovered", flush=True)
        return 1

    # Step 2: download replays via API
    print(f"\ndownloading {len(entries)} replays via API …", flush=True)
    n_new = 0
    for e in entries:
        rank, name, ep_id = e["rank"], e["team"], e["episode_id"]
        slug = slugify(name)
        team_dir = out_dir / slug
        episode_dir = team_dir / str(ep_id)
        replay_path = episode_dir / "replay.json"

        # Dedup across hours within the same day.
        if replay_path.exists() and replay_path.stat().st_size > 10_000:
            print(f"[rank {rank}] {name}  ep={ep_id}  (already downloaded)", flush=True)
            continue

        print(f"[rank {rank}] {name}  ep={ep_id} …", flush=True)
        replay = fetch_replay(ep_id, auth)
        if replay is None:
            continue

        episode_dir.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(json.dumps(replay), encoding="utf-8")

        info = replay.get("info") or {}
        meta = {
            "episode_id": ep_id,
            "team_seen_by": name,
            "rank_at_scrape": rank,
            "scrape_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "TeamNames": info.get("TeamNames"),
            "Agents": info.get("Agents"),
            "rewards": replay.get("rewards"),
            "statuses": replay.get("statuses"),
            "configuration": replay.get("configuration"),
            "list_episodes_entry": {
                "create_time": e.get("create_time"),
                "end_time": e.get("end_time"),
                "state": e.get("state"),
                "agents": e.get("agents"),
            },
        }
        (episode_dir / "episode.json").write_text(json.dumps(meta, indent=2),
                                                  encoding="utf-8")

        if not args.no_player:
            # play.html is 2 dirs deep from out_dir → assets at ../../player_assets
            wrap = make_offline_player(episode_dir, ep_id, replay_path,
                                       assets_rel="../../player_assets")
            print(f"  saved {episode_dir.relative_to(out_dir)}/replay.json + {wrap.name}",
                  flush=True)
        else:
            print(f"  saved {episode_dir.relative_to(out_dir)}/replay.json", flush=True)
        n_new += 1

    print(f"done — {n_new} new replay(s), {len(entries)-n_new} already on disk",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
