import numpy as np
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from movementpredictor.cnn.inferencing import get_bounding_box_info


#  scaling_factor = slope * y + intercept
def calculate_camera_angle(dataset: Dataset, pixel_per_axis):
    y_centers = []
    bbox_height_width = []

    for i, (x, target, ts, id) in enumerate(dataset):
        if i > 10000:
            break
        [[x_min, y_min], [x_max, y_max]] = get_bounding_box_info(x)
        width = (x_max - x_min) / pixel_per_axis
        height = (y_max - y_min) / pixel_per_axis

        bbox_height_width.append(width+height)
        y_centers.append((y_max+y_min)/(2*pixel_per_axis))

    model = LinearRegression()
    model.fit(np.array(y_centers).reshape(-1, 1), bbox_height_width)
    slope = model.coef_[0]
    intercept = model.intercept_

    return slope, intercept