import os 
import sys
import cv2


ROOT = os.path.abspath(
	os.path.join(os.path.dirname(__file__), "..")
	
)


if ROOT not in sys.path:
    sys.path.insert(0,ROOT)
    
    
from camera.CameraController import CameraController
from processing.image_analyzer import ImageAnalyzer


cam = CameraController()
analyzer = ImageAnalyzer()


exposures = [20000, 30000, 40000]


try:
	for exposure in exposures:
		cam.set_parameters(exposure=exposure)
		
		img = cam.capture_image()
		
		if img is None:
			print("FAILED:", exposure)
			continue
			
			
		features = analyzer.extract_features(img)
		proxy = analyzer.compute_defect_probability(features)
		
		
		filename = f"baseline_{exposure}.png"
		cv2.imwrite(filename, img)
		
		print("\nExposure:", exposure)
		print("Brightness:", features["brightness"])
		print("Contrast:", features["contrast"])
		print("Sharpness:", features["sharpness"])
		print("Proxy:", proxy)
		print("Saved:", filename)
		
		
finally:
 	cam.close()
 	
 	
 			
			
			
