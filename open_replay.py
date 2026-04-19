import json

json_path = 'replay.json'

with open(json_path, "r") as f:
    data = json.load(f)

print(data)