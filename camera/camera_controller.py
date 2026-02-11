import PySpin
import numpy as np


class camera_controller:

    def __init__(self):
        self.system = PySpin.system.GetInstance()
        self.cam_list = self.system.GetCameras()

        if self.cam_list.GetSize() == 0:
            self.system.ReleaseInstance()
            raise Exception("No camera detected")
        
        self.cam = self.cam_list.GetByIndex(0)
        self.cam.Init()
        self.cam.BeginAcquisition()
        print("camera started")

    
    def capture_image(self):
        image = self.cam.GetNextImage()
        
        if image.IsIncomplete():
            print("image incomplete")
            return None
        
        img_array = image.GetNDArray()
        image.Release()
        return img_array
    

    def self_parameter(self, exposure = None, gain = None):
        if exposure is not None:
            exposure_auto = self.cam.ExposureAuto.GetAccessMode()
            if exposure_auto == PySpin.RW:
                self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            self.cam.ExposureTime.SetValue(exposure)
        if gain is not None:
            self.cam.Gain.SetValue(gain)



    def close(self):
        try:
            self.cam.EndAcquisition()
            self.cam.DeInit()
            del self.cam
            self.cam_list.clear()
            self.system.ReleaseInstance()
            print("camera closed")
        except:
            print("error")

    

    
