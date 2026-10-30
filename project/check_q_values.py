import csv

CSV_FILE = (
    "experiment_data/"
    "lighting_change_run_02/"
    "evaluation/"
    "evaluation_results.csv"
)

seen = set()

with open(CSV_FILE, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        state = row["state"]

        if state in seen:
            continue

        seen.add(state)

        print()
        print("State:", state)
        print("Selected exposure:", row["selected_exposure"])

        for exposure in [500, 1000, 2000, 5000, 10000]:
            print(
                f"{exposure:5d} us -> "
                f"Q = {float(row[f'q_{exposure}']):.6f}"
            )
