import os
import sys
import csv
import cv2
import numpy as np

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from camera.CameraController import CameraController
from processing.image_analyzer import ImageAnalyzer


EXPOSURES = [20000, 30000, 40000]
SAMPLES_PER_EXPOSURE = 10

CSV_FILE = "baseline_results.csv"

IMAGE_DIR = "experiment_data/baseline"
os.makedirs(IMAGE_DIR, exist_ok=True)



def main():
    camera = CameraController()
    analyzer = ImageAnalyzer()

    rows = []

    try:
        for exposure in EXPOSURES:
        
            exposure_dir = os.path.join(IMAGE_DIR, str(exposure))
            os.makedirs(exposure_dir, exist_ok = True)

            print(f"\n=== Exposure: {exposure} us ===")

            camera.set_parameters(exposure=exposure)

            # exposure 변경 직후 첫 프레임은 버림
            camera.capture_image()

            for sample in range(SAMPLES_PER_EXPOSURE):

                image = camera.capture_image()

                if image is None:
                    print("Capture failed")
                    continue
                    
                cv2.imwrite(
                os.path.join(
                exposure_dir,
                f"image_{sample + 1:03d}.png"
       		),
       		image
                )

                features = analyzer.extract_features(image)

                proxy = analyzer.compute_defect_probability(
                    features
                )

                row = {
                    "exposure": exposure,
                    "sample": sample + 1,
                    "brightness": features["brightness"],
                    "contrast": features["contrast"],
                    "sharpness": features["sharpness"],
                    "proxy": proxy
                }

                rows.append(row)

                print(
                    f"{sample + 1:02d} | "
                    f"B={features['brightness']:.2f} "
                    f"C={features['contrast']:.2f} "
                    f"S={features['sharpness']:.2f} "
                    f"Proxy={proxy:.4f}"
                )

                # exposure별 대표 이미지 한 장 저장
                if sample == 0:
                    cv2.imwrite(
                        f"baseline_{exposure}.png",
                        image
                    )

        # CSV 저장
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "exposure",
                    "sample",
                    "brightness",
                    "contrast",
                    "sharpness",
                    "proxy"
                ]
            )

            writer.writeheader()
            writer.writerows(rows)

        print("\n=== SUMMARY ===")

        for exposure in EXPOSURES:
            exposure_rows = [
                r for r in rows
                if r["exposure"] == exposure
            ]

            if not exposure_rows:
                continue

            print(f"\nExposure {exposure} us")

            for key in [
                "brightness",
                "contrast",
                "sharpness",
                "proxy"
            ]:
                values = [
                    r[key] for r in exposure_rows
                ]

                print(
                    f"{key}: "
                    f"mean={np.mean(values):.4f}, "
                    f"std={np.std(values):.4f}"
                )

        print(f"\nSaved: {CSV_FILE}")

    finally:
        camera.close()


if __name__ == "__main__":
    main()
