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

EXPERIMENT_NAME = "defect_run_01"

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


EXPOSURES = [
    20000,
    30000,
    40000
]


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

        "reward":
            "(before_proxy - after_proxy) * 10"
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

    try:

        save_config(env)

        print("=== Training started ===")
        print("Experiment:", EXPERIMENT_NAME)
        print("Steps:", STEPS)
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

            # -----------------------------------------
            # Save images according to selected exposure
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

                # ------------------------
                # BEFORE features
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
                # AFTER features
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
                # Q values BEFORE decision
                # ------------------------

                "q_before_20000":
                    q_before[20000],

                "q_before_30000":
                    q_before[30000],

                "q_before_40000":
                    q_before[40000],

                # ------------------------
                # Q values AFTER learning
                # ------------------------

                "q_after_20000":
                    q_after[20000],

                "q_after_30000":
                    q_after[30000],

                "q_after_40000":
                    q_after[40000],

                "image_before":
                    before_path,

                "image_after":
                    after_path,
            }

            rows.append(row)

            print(
                f"Step {step:03d} | "
                f"State={result['state_key']} | "
                f"Exposure={exposure} | "
                f"Before={row['before_proxy']:.4f} | "
                f"After={row['after_proxy']:.4f} | "
                f"Reward={row['reward']:.4f}"
            )

        # -----------------------------------------
        # Save final learned Q-table
        # -----------------------------------------

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
