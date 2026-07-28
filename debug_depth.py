import numpy as np

data = np.load("./shape_reconstruction/calibration/sensor_118/depth_calibration/Pixel_to_Depth.npy")

print(data)
print(data.shape)
print(data.dtype)