import os
import sys
import csv
import numpy as np
import cv2


ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from environment.camera_environment import CameraEnvironment


STEPS = 30

EXPERIMENT_NAME = "corrected_rl_run_01"

EXPERIMENT_DIR = os.path.join(
    ROOT,
    "experiment_data",
    EXPERIMENT_NAME
)

EVALUATION_DIR = os.path.join(
    EXPERIMENT_DIR,
    "evaluation"
)

Q_TABLE_FILE = os.path.join(
    EXPERIMENT_DIR,
    "q_table.pkl"
)

CSV_FILE = os.path.join(
    EVALUATION_DIR,
    "evaluation_results.csv"
)


EXPOSURES = [
    20000,
    30000,
    40000
]


def create_directories():

    os.makedirs(
        EVALUATION_DIR,
        exist_ok=True
    )

    for exposure in EXPOSURES:

        exposure_dir = os.path.join(
            EVALUATION_DIR,
            str(exposure)
        )

        os.makedirs(
            exposure_dir,
            exist_ok=True
        )


def main():

    create_directories()

    env = CameraEnvironment()

    rows = []

    try:

        # -----------------------------------------
        # Load learned Q-table
        # -----------------------------------------

        env.agent.load_q_table(
            Q_TABLE_FILE
        )

        # Evaluation:
        # no random exploration
        env.agent.epsilon = 0.0

        print(
            "=== Evaluation started ==="
        )

        print(
            "Loaded states:",
            len(env.agent.q_table)
        )

        print(
            "Epsilon:",
            env.agent.epsilon
        )

        print()

        for _ in range(STEPS):

            result = env.step(
                training=False
            )

            if "error" in result:
                print(result)
                continue

            step = result["step"]

            exposure = result[
                "action"
            ]["exposure"]

            exposure_dir = os.path.join(
                EVALUATION_DIR,
                str(exposure)
            )

            # -----------------------------------------
            # Save images
            # -----------------------------------------

            before_path = os.path.join(
                exposure_dir,
                f"step_{step:03d}_before.png"
            )

            after_path = os.path.join(
                exposure_dir,
                f"step_{step:03d}_after.png"
            )

            cv2.imwrite(
                before_path,
                result["image_before"]
            )

            cv2.imwrite(
                after_path,
                result["image_after"]
            )

            state_key = result[
                "state_key"
            ]

            state_seen = (
                state_key
                in env.agent.q_table
            )

            q_values = result[
                "q_values_before"
            ]

            row = {

                "step":
                    step,

                "state_key":
                    str(state_key),

                "state_seen":
                    state_seen,

                "selected_exposure":
                    exposure,

                "before_proxy":
                    result[
                        "before_defect_prob"
                    ],

                "after_proxy":
                    result[
                        "after_defect_prob"
                    ],

                "reward":
                    result["reward"],

                # ------------------------
                # BEFORE
                # ------------------------

                "brightness_before":
                    result[
                        "features_before"
                    ]["brightness"],

                "contrast_before":
                    result[
                        "features_before"
                    ]["contrast"],

                "sharpness_before":
                    result[
                        "features_before"
                    ]["sharpness"],

                # ------------------------
                # AFTER
                # ------------------------

                "brightness_after":
                    result[
                        "features_after"
                    ]["brightness"],

                "contrast_after":
                    result[
                        "features_after"
                    ]["contrast"],

                "sharpness_after":
                    result[
                        "features_after"
                    ]["sharpness"],

                # ------------------------
                # Q-table values used
                # for this decision
                # ------------------------

                "q_20000":
                    q_values[20000],

                "q_30000":
                    q_values[30000],

                "q_40000":
                    q_values[40000],

                "image_before":
                    before_path,

                "image_after":
                    after_path,
            }

            rows.append(row)

            print(
                f"Step {step:03d} | "
                f"State={state_key} | "
                f"Exposure={exposure} | "
                f"Before={row['before_proxy']:.4f} | "
                f"After={row['after_proxy']:.4f} | "
                f"Reward={row['reward']:.4f} | "
                f"Seen={state_seen}"
            )

        if not rows:
            print(
                "No successful evaluation samples."
            )
            return

        # -----------------------------------------
        # Save CSV
        # -----------------------------------------

        with open(
            CSV_FILE,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(rows)

        # -----------------------------------------
        # Summary
        # -----------------------------------------

        after_values = [
            row["after_proxy"]
            for row in rows
        ]

        rewards = [
            row["reward"]
            for row in rows
        ]

        seen_count = sum(
            1
            for row in rows
            if row["state_seen"]
        )

        exposure_counts = {
            exposure: 0
            for exposure in EXPOSURES
        }

        for row in rows:

            exposure_counts[
                row["selected_exposure"]
            ] += 1

        print()
        print(
            "=== Evaluation Summary ==="
        )

        print(
            "Mean after proxy:",
            round(
                float(
                    np.mean(after_values)
                ),
                4
            )
        )

        print(
            "Std after proxy:",
            round(
                float(
                    np.std(after_values)
                ),
                4
            )
        )

        print(
            "Mean reward:",
            round(
                float(
                    np.mean(rewards)
                ),
                4
            )
        )

        print(
            f"Known states: "
            f"{seen_count}/{len(rows)}"
        )

        print()
        print(
            "Exposure selection counts:"
        )

        for exposure in EXPOSURES:

            print(
                f"{exposure}: "
                f"{exposure_counts[exposure]}"
            )

        print()
        print(
            "Saved:",
            CSV_FILE
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()