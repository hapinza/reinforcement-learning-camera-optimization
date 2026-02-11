import cv2
import numpy as np

#turn the original image into what camera can understand
class ImageAnalyzer:
    def __init__(self):
        print("image analyzer is initilaized")

    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image


        brightness = float(np.mean(gray))

        contrast = float(np.std(gray))

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        features = {
            "brightness" : brightness,
            "contrast" : contrast,
            "sharpness": sharpness
        }


        return features
    

    def compute_defect_probability(self, features):
        brightness = features["brightness"]
        contrast = features["contrast"]
        sharpness = features["sharpness"]

        brightness_score = max(0, 1 - abs(brightness - 125)/125)
        contrast_score = min(1, contrast/80)
        sharpness_score = min(1, sharpness/ 1000)


        quality_score = (brightness_score + contrast_score + sharpness_score) / 3
        defect_prob = 1 - quality_score
        return defect_prob
    
