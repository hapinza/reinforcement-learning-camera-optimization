import PySpin

class camera_controller:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()

        if self.cam_list.GetSize() == 0:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            raise Exception("No camera detected")

        self.cam = self.cam_list.GetByIndex(0)
        self.cam.Init()

        # Auto exposure off
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)

        # Mono8
        self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)

        # Manual stream buffer = 3
        stream_map = self.cam.GetTLStreamNodeMap()

        mode = PySpin.CEnumerationPtr(
            stream_map.GetNode("StreamBufferCountMode")
        )
        manual_entry = mode.GetEntryByName("Manual")
        mode.SetIntValue(manual_entry.GetValue())

        count = PySpin.CIntegerPtr(
            stream_map.GetNode("StreamBufferCountManual")
        )
        count.SetValue(3)

        # Single frame
        self.cam.AcquisitionMode.SetValue(
            PySpin.AcquisitionMode_SingleFrame
        )

        print("camera initialized")

    def capture_image(self):
        try:
            self.cam.BeginAcquisition()

            image = self.cam.GetNextImage(10000)

            if image.IsIncomplete():
                print("image incomplete")
                image.Release()
                self.cam.EndAcquisition()
                return None

            img_array = image.GetNDArray().copy()

            image.Release()
            self.cam.EndAcquisition()

            return img_array

        except Exception as e:
            print("capture error:", e)

            try:
                self.cam.EndAcquisition()
            except:
                pass

            return None

    def set_parameter(self, exposure=None, gain=None):
        if exposure is not None:
            exposure = max(
                self.cam.ExposureTime.GetMin(),
                min(exposure, self.cam.ExposureTime.GetMax())
            )
            self.cam.ExposureTime.SetValue(exposure)

        if gain is not None:
            gain = max(
                self.cam.Gain.GetMin(),
                min(gain, self.cam.Gain.GetMax())
            )
            self.cam.Gain.SetValue(gain)

    def close(self):
        try:
            self.cam.DeInit()
            del self.cam
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            print("camera closed")
        except Exception as e:
            print("close error:", e)
