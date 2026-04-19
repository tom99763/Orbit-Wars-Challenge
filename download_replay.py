import requests

#when you click the leaderboard button, the epsisode id is in the end of url: https://www.kaggle.com/competitions/orbit-wars/leaderboard?submissionId=51799179&episodeId=75149536

import json, pathlib
# Read credentials from ~/.kaggle/kaggle.json (do NOT commit the key itself)
_cfg = json.loads((pathlib.Path.home() / ".kaggle" / "kaggle.json").read_text())
username = _cfg["username"]
key = _cfg["key"]
episode_id = '75149536'

# Replay
url = f"https://www.kaggle.com/api/v1/competitions/episodes/{episode_id}/replay"
r = requests.get(url, auth=(username, key))
with open("replay.json", "wb") as f:
    f.write(r.content)
print("replay.json")
print(r.content)