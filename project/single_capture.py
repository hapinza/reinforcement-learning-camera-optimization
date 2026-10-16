import PySpin

system = PySpin.System.GetInstance()
cam_list = system.GetCameras()

print("Number of cameras detected:", cam_list.GetSize())

if cam_list.GetSize() == 0:
    cam_list.Clear()
    system.ReleaseInstance()
    raise RuntimeError("No camera detected")

cam = cam_list.GetByIndex(0)

try:
    cam.Init()

    nodemap = cam.GetNodeMap()

    acquisition_mode = PySpin.CEnumerationPtr(
        nodemap.GetNode("AcquisitionMode")
    )

    single_frame = acquisition_mode.GetEntryByName("SingleFrame")
    acquisition_mode.SetIntValue(single_frame.GetValue())

    cam.BeginAcquisition()

    image = cam.GetNextImage(5000)

    if image.IsIncomplete():
        print("Image incomplete:", image.GetImageStatus())
    else:
        print("Image acquired")
        print("Width:", image.GetWidth())
        print("Height:", image.GetHeight())

        image.Save("single_test.jpg")
        print("Saved as single_test.jpg")

    image.Release()
    cam.EndAcquisition()
    cam.DeInit()

finally:
    del cam
    cam_list.Clear()
    system.ReleaseInstance()
