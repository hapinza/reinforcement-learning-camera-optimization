import os
import csv
import statistics
from collections import defaultdict

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

EXPERIMENT_NAME = "defect_run_01"

CSV_FILE = os.path.join(
    ROOT,
    "experiment_data",
    EXPERIMENT_NAME,
    "baseline_results.csv"
)

groups = defaultdict(list)

with open(CSV_FILE, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        exposure = int(row["exposure"])
        proxy = float(row["proxy"])

        groups[exposure].append(proxy)

print("\n=== Baseline Exposure Analysis ===")
print("Exposure | Samples | Mean Proxy | Std Proxy")
print("------------------------------------------------")

for exposure, values in sorted(groups.items()):
    mean = statistics.mean(values)
    std = statistics.pstdev(values)

    print(
        f"{exposure:8d} | "
        f"{len(values):7d} | "
        f"{mean:.4f}     | "
        f"{std:.4f}"
    )

best_exposure = min(
    groups,
    key=lambda e: statistics.mean(groups[e])
)

print()
print(f"Best baseline exposure: {best_exposure} us")
print(
    f"Best baseline mean proxy: "
    f"{statistics.mean(groups[best_exposure]):.4f}"
)
