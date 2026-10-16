import os
import csv
from collections import defaultdict

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

EXPERIMENT_NAME = "fine_pattern_run_01"

CSV_FILE = os.path.join(
    ROOT,
    "experiment_data",
    EXPERIMENT_NAME,
    "training",
    "training_results.csv"
)

groups = defaultdict(list)

with open(CSV_FILE, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        exposure = int(row["selected_exposure"])
        proxy = float(row["after_proxy"])

        groups[exposure].append(proxy)

print("\n=== Exposure Analysis ===")
print("Exposure | Samples | Mean Proxy")
print("--------------------------------")

for exposure in sorted(groups):
    values = groups[exposure]
    mean_proxy = sum(values) / len(values)

    print(
        f"{exposure:8d} | "
        f"{len(values):7d} | "
        f"{mean_proxy:.4f}"
    )

best_exposure = min(
    groups,
    key=lambda e: sum(groups[e]) / len(groups[e])
)

print()
print(f"Lowest mean proxy: {best_exposure} us")
