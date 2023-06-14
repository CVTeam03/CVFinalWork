import numpy as np
import cv2
from scipy import ndimage


def convolve(image, kernel):
    return ndimage.filters.convolve(image, kernel, mode='constant', cval=0)


def to_gray(image):
    if len(image.shape) == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def disparity(image_left, image_right, kernel=7, search_depth=30):
    gray_left = to_gray(image_left)
    gray_right = to_gray(image_right)
    kernel = np.ones((kernel, kernel), dtype=np.float32)

    min_ssd = np.full(gray_left.shape, float('inf'), dtype=np.float32)
    labels = np.zeros(gray_left.shape, dtype=int)
    for offset in range(search_depth):
        shifted = gray_right if offset == 0 else gray_right[:, :-offset]
        raw_ssd = np.square(gray_left[:, offset:] - shifted)
        ssd = convolve(raw_ssd, kernel)
        label_min = ssd < min_ssd[:, offset:]
        min_ssd[:, offset:][label_min] = ssd[label_min]
        labels[:, offset:][label_min] = offset

    return labels
