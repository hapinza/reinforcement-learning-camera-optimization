import os
import sys
import cv2



ROOT = os.path.abspath(
	os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
	sys.path.insert(0,ROOT)
	

from camera.CameraController import CameraController

cam = CameraController()


try:
    img = cam.capture_image()
    
    if img is not None:
    	path = "test_snapshot.png"
    	
    	cv2.imwrite(path, img)
    	
    	
    	print("snapshot success")
    	print("shape:", img.shape)
    	print("saved:", path)
    	
    	
    else:
    	print("snapshot failed")
    	
    	
finally:
    cam.close()
    
    
    
    
        		
