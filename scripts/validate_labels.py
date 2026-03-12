"""Quick validation of labeling output."""
import csv
from collections import defaultdict

with open("local_artifacts/labeled_grid-00000-of-00001.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total labeled rows: {len(rows)}")

with_target = [r for r in rows if r["time_to_next_train"] and r["time_to_next_train"] != "None"]
without_target = [r for r in rows if not r["time_to_next_train"] or r["time_to_next_train"] == "None"]
print(f"Rows WITH time_to_next_train: {len(with_target)}")
print(f"Rows with NULL target (timer flush): {len(without_target)}")

with_feature = [r for r in rows if r["minutes_since_last_train"] and r["minutes_since_last_train"] != "None"]
print(f"Rows WITH minutes_since_last_train: {len(with_feature)}")

# Show countdown sequences
node_series = defaultdict(list)
for r in with_target:
    node_series[r["node_id"]].append(
        (r["snapshot_time"], float(r["time_to_next_train"]), int(r["train_present"]))
    )

print(f"\nNodes with labeled data: {len(node_series)}")
print("\n--- Sample countdown sequences ---")
count = 0
for nid, series in sorted(node_series.items()):
    series.sort()
    if len(series) >= 2:
        print(f"\n{nid}:")
        for ts, ttnt, tp in series:
            print(f"  {ts} | present={tp} | time_to_next={ttnt}")
        count += 1
        if count >= 4:
            break

# Verify countdown decrements by 1 each minute
print("\n--- Validation: countdown should decrement by 1.0 per minute ---")
errors = 0
for nid, series in node_series.items():
    series.sort()
    for i in range(1, len(series)):
        if series[i][0] > series[i - 1][0]:
            expected = series[i - 1][1] - 1.0
            actual = series[i][1]
            if abs(actual - expected) > 0.15:
                errors += 1
                if errors <= 5:
                    print(
                        f"  ISSUE {nid}: {series[i-1][0]} ttnt={series[i-1][1]}"
                        f" -> {series[i][0]} ttnt={actual} (expected ~{expected})"
                    )
if errors == 0:
    print("  ALL GOOD - every countdown decrements by exactly 1.0/minute")
else:
    print(f"  Found {errors} sequence issues (may span multiple flush groups)")
