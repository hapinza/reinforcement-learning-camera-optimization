import os
import csv
import cv2

from camera.CameraController import CameraController
from processing.image_analyzer import ImageAnalyzer


EXPOSURES = [20000, 30000, 40000]
IMAGES_PER_EXPOSURE = 3

OUTPUT_DIR = "experiment_data/exposure_transition_test"
CSV_PATH = os.path.join(OUTPUT_DIR, "results.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    camera = CameraController()
    analyzer = ImageAnalyzer()

    results = []

    try:
        for exposure in EXPOSURES:

            exposure_dir = os.path.join(
                OUTPUT_DIR,
                str(exposure)
            )
            os.makedirs(exposure_dir, exist_ok=True)

            print(f"\n=== Set Exposure: {exposure} us ===")

            # Change camera exposure once.
            camera.set_parameters(exposure=exposure)

            # Capture 3 consecutive images after the change.
            for frame_number in range(1, IMAGES_PER_EXPOSURE + 1):

                image = camera.capture_image()

                if image is None:
                    print(
                        f"Frame {frame_number}: capture failed"
                    )
                    continue

                features = analyzer.extract_features(image)
                proxy = analyzer.compute_defect_probability(features)

                image_path = os.path.join(
                    exposure_dir,
                    f"image_{frame_number:02d}.png"
                )

                cv2.imwrite(image_path, image)

                row = {
                    "exposure": exposure,
                    "frame": frame_number,
                    "brightness": features["brightness"],
                    "contrast": features["contrast"],
                    "sharpness": features["sharpness"],
                    "proxy": proxy,
                    "image_path": image_path,
                }

                results.append(row)

                print(
                    f"Frame {frame_number} | "
                    f"B={features['brightness']:.2f} | "
                    f"C={features['contrast']:.2f} | "
                    f"S={features['sharpness']:.2f} | "
                    f"Proxy={proxy:.4f}"
                )

    finally:
        camera.close()

    with open(CSV_PATH, "w", newline="") as csv_file:
        fieldnames = [
            "exposure",
            "frame",
            "brightness",
            "contrast",
            "sharpness",
            "proxy",
            "image_path",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n=== Experiment Complete ===")
    print(f"Images saved to: {OUTPUT_DIR}")
    print(f"Results saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()