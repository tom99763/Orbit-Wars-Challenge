"""Scrape top-N team replay animations from Orbit Wars leaderboard.

For each of the current top-N teams we:
  1. Open the "episodes" dialog from their leaderboard row (live_tv icon).
  2. Click the first (most recent) episode to start the player.
  3. Capture the JSON response of `competitions.EpisodeService/GetEpisodeReplay`
     — this holds the full game replay (per-step state).
  4. Capture episode metadata from `GetEpisode`.
  5. Save both JSONs + a screenshot of the player + the full page HTML.

Also (once per run) download the Kaggle iframe visualizer bundle and emit a
self-contained `player.html` that loads a given replay locally.

Default output layout: simulation/<YYYY-MM-DD>/<HH-MM>/
  leaderboard.html / .png
  rank0X_<team>/
      page.html
      player.png
      episode_<id>.json        (metadata)
      replay_<id>.json         (full per-step replay)
      play_<id>.html           (offline wrapper — open in a browser)
  player_assets/               (shared visualizer bundle)

Override with --out-dir to pin to a specific folder (used by pipeline.py so
the scrape and the parser share one timestamp).
"""

import argparse
import datetime
import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request

from playwright.sync_api import sync_playwright

LEADERBOARD_URL = "https://www.kaggle.com/competitions/orbit-wars/leaderboard"
VISUALIZER_INDEX = "https://www.kaggleusercontent.com/episode-visualizers/orbit_wars/default/index.html"
TOP_N = 10

REPLAY_EP_RE = re.compile(r"EpisodeService/(GetEpisodeReplay|GetEpisode)\b")


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


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


def click_first_episode(page) -> str:
    return page.evaluate(
        """() => {
            const modal = document.querySelector('.MuiModal-root');
            if (!modal) return 'no_modal';
            const lis = modal.querySelectorAll('li');
            for (const li of lis) {
                const t = (li.innerText || '').toLowerCase();
                if (t.includes('ago')) { li.click(); return 'clicked:' + t.slice(0, 40); }
            }
            return 'no_episode';
        }"""
    )


def close_dialog(page) -> None:
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(1000)
    except Exception:
        pass


def download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())


def fetch_visualizer_bundle(assets_dir: pathlib.Path) -> pathlib.Path:
    """Download the iframe's index.html plus any JS/CSS it references.

    Returns the local path to index.html.
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    index_path = assets_dir / "index.html"
    download(VISUALIZER_INDEX, index_path)
    html = index_path.read_text(encoding="utf-8", errors="replace")
    # Pull in any relative ./assets/* references
    for m in re.finditer(r'(?:src|href)="(\.\/[^"#?]+|assets\/[^"#?]+)"', html):
        rel = m.group(1).lstrip("./")
        url = f"https://www.kaggleusercontent.com/episode-visualizers/orbit_wars/default/{rel}"
        local = assets_dir / rel
        try:
            download(url, local)
        except Exception as e:
            print(f"  (warn) could not fetch {url}: {e}", flush=True)
    return index_path


def make_offline_player(team_dir: pathlib.Path, episode_id: int, replay_path: pathlib.Path,
                       assets_rel: str) -> pathlib.Path:
    """Wrapper HTML: iframe loads visualizer, parent postMessages the replay in."""
    out = team_dir / f"play_{episode_id}.html"
    replay_rel = replay_path.name
    out.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Episode {episode_id}</title>
<style>html,body{{margin:0;height:100%;background:#000;color:#eee;font-family:sans-serif}}
iframe{{width:100%;height:100vh;border:0}}</style></head><body>
<iframe id="player" src="{assets_rel}/index.html"></iframe>
<script>
  // Kaggle visualizers receive replay JSON via postMessage as an "environment" object.
  // Schema: {{ type: "update", environment: <replayJson> }}
  fetch({json.dumps(replay_rel)})
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Default: simulation/<YYYY-MM-DD>/<HH-MM>/")
    ap.add_argument("--top-n", type=int, default=TOP_N)
    args = ap.parse_args()

    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir)
    else:
        now = datetime.datetime.now()
        out_dir = (pathlib.Path(__file__).parent / "simulation"
                   / now.date().isoformat() / now.strftime("%H-%M"))
    out_dir.mkdir(parents=True, exist_ok=True)
    top_n = args.top_n
    print(f"out_dir: {out_dir}  top_n: {top_n}", flush=True)

    print("fetching visualizer bundle …", flush=True)
    assets_dir = out_dir / "player_assets"
    fetch_visualizer_bundle(assets_dir)
    print(f"  bundle → {assets_dir}", flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 1600, "height": 1400},
            user_agent=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
            bypass_csp=True,
        )
        page = context.new_page()

        captured: dict = {"replay": None, "meta": None}

        def on_response(resp):
            m = REPLAY_EP_RE.search(resp.url)
            if not m:
                return
            try:
                body = resp.json()
            except Exception:
                return
            if m.group(1) == "GetEpisodeReplay":
                captured["replay"] = body
            else:
                captured["meta"] = body

        page.on("response", on_response)

        print(f"loading {LEADERBOARD_URL}", flush=True)
        page.goto(LEADERBOARD_URL, wait_until="domcontentloaded", timeout=60000)
        wait_for(page, "Shun_PI", timeout_ms=90000)
        page.wait_for_timeout(3000)

        (out_dir / "leaderboard.html").write_text(page.content(), encoding="utf-8")
        page.screenshot(path=str(out_dir / "leaderboard.png"), full_page=True)

        top = get_top_n_teams(page, top_n)
        print(f"top {top_n}: {top}", flush=True)

        for rank, name in enumerate(top, start=1):
            slug = f"rank{rank:02d}_{slugify(name)}"
            team_dir = out_dir / slug
            team_dir.mkdir(parents=True, exist_ok=True)
            print(f"[rank {rank}] {name}", flush=True)

            # Reset captures
            captured["replay"] = None
            captured["meta"] = None

            res = click_live_tv(page, name)
            print(f"  live_tv click: {res}", flush=True)
            page.wait_for_timeout(4000)
            # Wait for dialog modal
            try:
                page.wait_for_selector(".MuiModal-root", timeout=15000)
            except Exception:
                print("  WARN: modal not seen", flush=True)

            ep_res = click_first_episode(page)
            print(f"  episode click: {ep_res}", flush=True)

            # Wait for replay JSON to arrive (up to 30s)
            deadline = time.time() + 30
            while time.time() < deadline and captured["replay"] is None:
                page.wait_for_timeout(500)

            (team_dir / "page.html").write_text(page.content(), encoding="utf-8")
            page.screenshot(path=str(team_dir / "player.png"), full_page=True)

            meta = captured["meta"] or {}
            replay = captured["replay"]
            if replay is None:
                print("  WARN: no replay captured", flush=True)
                close_dialog(page)
                continue

            ep_id = (meta.get("episode") or {}).get("id") or meta.get("episodeId") or int(time.time())
            (team_dir / f"episode_{ep_id}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8")
            replay_path = team_dir / f"replay_{ep_id}.json"
            replay_path.write_text(json.dumps(replay), encoding="utf-8")
            wrap = make_offline_player(team_dir, ep_id, replay_path, assets_rel="../player_assets")
            print(f"  saved replay_{ep_id}.json + {wrap.name}", flush=True)
            close_dialog(page)

        browser.close()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
