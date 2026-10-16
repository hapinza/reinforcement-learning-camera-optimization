import os
import sys
import csv
import json
import cv2


ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from environment.camera_environment import CameraEnvironment


STEPS = 300

EXPERIMENT_NAME = "fine_pattern_run_01"

EXPOSURES = [
    500,
    1000,
    2000,
    5000,
    10000,
]

EXPERIMENT_DIR = os.path.join(
    ROOT,
    "experiment_data",
    EXPERIMENT_NAME
)

TRAINING_DIR = os.path.join(
    EXPERIMENT_DIR,
    "training"
)

CSV_FILE = os.path.join(
    TRAINING_DIR,
    "training_results.csv"
)

Q_TABLE_FILE = os.path.join(
    EXPERIMENT_DIR,
    "q_table.pkl"
)

CONFIG_FILE = os.path.join(
    EXPERIMENT_DIR,
    "config.json"
)


def create_directories():
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)

    for exposure in EXPOSURES:
        exposure_dir = os.path.join(
            TRAINING_DIR,
            str(exposure)
        )

        os.makedirs(
            exposure_dir,
            exist_ok=True
        )


def save_config(env):
    config = {
        "experiment": EXPERIMENT_NAME,
        "steps": STEPS,
        "actions": EXPOSURES,
        "learning_rate": env.agent.lr,
        "gamma": env.agent.gamma,
        "epsilon": env.agent.epsilon,
        "stale_frame_discard": True,
        "reward": "(before_proxy - after_proxy) * 10"
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(
            config,
            f,
            indent=4
        )


def main():
    create_directories()

    env = CameraEnvironment()
    rows = []

    expected_actions = [
        action["exposure"]
        for action in env.actions
    ]

    if expected_actions != EXPOSURES:
        raise ValueError(
            f"Exposure mismatch: "
            f"CameraEnvironment={expected_actions}, "
            f"run_training={EXPOSURES}"
        )

    try:
        save_config(env)

        print("=== Training started ===")
        print("Experiment:", EXPERIMENT_NAME)
        print("Steps:", STEPS)
        print("Actions:", EXPOSURES)
        print()

        for _ in range(STEPS):

            result = env.step(
                training=True
            )

            if "error" in result:
                print(result)
                continue

            step = result["step"]

            exposure = result[
                "action"
            ]["exposure"]

            exposure_dir = os.path.join(
                TRAINING_DIR,
                str(exposure)
            )

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

            q_before = result[
                "q_values_before"
            ]

            q_after = result[
                "q_values_after"
            ]

            row = {
                "step":
                    step,

                "state_key":
                    str(
                        result["state_key"]
                    ),

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

                "image_before":
                    before_path,

                "image_after":
                    after_path,
            }

            # Save Q-values dynamically for all exposures
            for exposure_value in EXPOSURES:
                row[
                    f"q_before_{exposure_value}"
                ] = q_before[exposure_value]

                row[
                    f"q_after_{exposure_value}"
                ] = q_after[exposure_value]

            rows.append(row)

            print(
                f"Step {step:03d} | "
                f"State={result['state_key']} | "
                f"Exposure={exposure} | "
                f"Before={row['before_proxy']:.4f} | "
                f"After={row['after_proxy']:.4f} | "
                f"Reward={row['reward']:.4f}"
            )

        env.agent.save_q_table(
            Q_TABLE_FILE
        )

        if rows:

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

        print()
        print("=== Training complete ===")

        print(
            "Learned states:",
            len(env.agent.q_table)
        )

        print(
            "Q-table:",
            Q_TABLE_FILE
        )

        print(
            "CSV:",
            CSV_FILE
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
