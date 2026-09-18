import os
import sys
import csv
import cv2

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from environment.camera_environment import CameraEnvironment


STEPS = 300
CSV_FILE = "training_results.csv"
Q_TABLE_FILE = "q_table.pkl"


IMAGE_DIR = "experiment_data/training"
os.makedirs(IMAGE_DIR, exist_ok=True)

def main():
    env = CameraEnvironment()
    rows = []

    try:
        for _ in range(STEPS):
            result = env.step(training=True)


            if "error" in result:
                print(result)
                continue
                
            step = result["step"]
            
            cv2.imwrite(
            os.path.join(IMAGE_DIR, f"step_{step:03d}_before.png"),
            result["image_before"]
            )
            
            cv2.imwrite(
            os.path.join(IMAGE_DIR, f"step_{step:03d}_after.png"),
            result["image_after"]
            )

            row = {
                "step": result["step"],
                "exposure": result["action"]["exposure"],
                "before_proxy": result["before_defect_prob"],
                "after_proxy": result["after_defect_prob"],
                "reward": result["reward"],
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
            }

            rows.append(row)

            print(
                f"Step {row['step']:03d} | "
                f"Exposure={row['exposure']} | "
                f"Reward={row['reward']:.4f}"
            )

        env.agent.save_q_table(Q_TABLE_FILE)

        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

        print()
        print("Training complete")
        print("Learned states:", len(env.agent.q_table))
        print("Saved:", Q_TABLE_FILE)
        print("Saved:", CSV_FILE)

    finally:
        env.close()


if __name__ == "__main__":
    main()
