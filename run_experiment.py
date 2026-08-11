from environment.camera_environment import CameraEnvironment

env = CameraEnvironment()

try:
    for _ in range(30):
        result = env.step()
        print(result)
finally:
    env.close()
