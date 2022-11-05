import cv2
import numpy as np


class BGRtoRGB(object):
    """bgr format to rgb"""

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class DepthScale(object):
    """scale depth to meters"""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, depth):
        depth = depth.astype(np.float32)
        return depth * self.scale


class DepthFilter(object):
    """scale depth to meters"""

    def __init__(self, max_depth):
        self.max_depth = max_depth

    def __call__(self, depth):
        far_mask = depth > self.max_depth
        depth[far_mask] = 0.
        return depth


class Undistort(object):
    """scale depth to meters"""

    def __init__(self,
                 w, h,
                 fx, fy, cx, cy,
                 k1, k2, k3, k4, k5, k6,
                 p1, p2,
                 interpolation):
        self.interpolation = interpolation
        K = np.array([[fx, 0., cx],
                      [0., fy, cy],
                      [0., 0., 1.]])

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            K,
            np.array([k1, k2, p1, p2, k3, k4, k5, k6]),
            np.eye(3),
            K,
            (w, h),
            cv2.CV_32FC1)

    def __call__(self, im):
        im = cv2.remap(im, self.map1x, self.map1y, self.interpolation)
        return im
