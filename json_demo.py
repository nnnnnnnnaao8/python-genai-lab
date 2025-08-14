import json
stats = {"count": 5, "mean": 12.3}
with open("artifacts/demo_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
with open("artifacts/demo_stats.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)
print("SAVED:", stats)
print("LOADED:", loaded)

