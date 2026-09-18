import os
import sys
import csv
import numpy as np

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from environment.camera_environment import CameraEnvironment


STEPS = 30

Q_TABLE_FILE = os.path.join(ROOT, "q_table.pkl")
CSV_FILE = os.path.join(ROOT, "evaluation_results.csv")


def main():
    env = CameraEnvironment()
    rows = []

    try:
        # Load learned Q-table
        env.agent.load_q_table(Q_TABLE_FILE)

        # Evaluation = no exploration
        env.agent.epsilon = 0.0

        print("=== Evaluation started ===")
        print("Loaded states:", len(env.agent.q_table))
        print("Epsilon:", env.agent.epsilon)

        for _ in range(STEPS):

            result = env.step(training=False)

            if "error" in result:
                print(result)
                continue

            state_key = env.agent.state_to_key(
                result["features_before"]
            )

            state_seen = state_key in env.agent.q_table

            row = {
                "step": result["step"],
                "exposure": result["action"]["exposure"],

                "before_proxy":
                    result["before_defect_prob"],

                "after_proxy":
                    result["after_defect_prob"],

                "reward":
                    result["reward"],

                "brightness_before":
                    result["features_before"]["brightness"],

                "contrast_before":
                    result["features_before"]["contrast"],

                "sharpness_before":
                    result["features_before"]["sharpness"],

                "brightness_after":
                    result["features_after"]["brightness"],

                "contrast_after":
                    result["features_after"]["contrast"],

                "sharpness_after":
                    result["features_after"]["sharpness"],

                "state_key":
                    str(state_key),

                "state_seen":
                    state_seen,
            }

            rows.append(row)

            print(
                f"Step {row['step']:03d} | "
                f"Exposure={row['exposure']} | "
                f"Before={row['before_proxy']:.4f} | "
                f"After={row['after_proxy']:.4f} | "
                f"Reward={row['reward']:.4f} | "
                f"Seen={row['state_seen']}"
            )

        if not rows:
            print("No successful evaluation samples.")
            return

        with open(CSV_FILE, "w", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(rows)

        after_values = [
            row["after_proxy"]
            for row in rows
        ]

        rewards = [
            row["reward"]
            for row in rows
        ]

        seen_count = sum(
            1 for row in rows
            if row["state_seen"]
        )

        print("\n=== Evaluation Summary ===")

        print(
            "Mean after proxy:",
            round(float(np.mean(after_values)), 4)
        )

        print(
            "Std after proxy:",
            round(float(np.std(after_values)), 4)
        )

        print(
            "Mean reward:",
            round(float(np.mean(rewards)), 4)
        )

        print(
            f"Known states: {seen_count}/{len(rows)}"
        )

        print("Saved:", CSV_FILE)

    finally:
        env.close()


if __name__ == "__main__":
    main()