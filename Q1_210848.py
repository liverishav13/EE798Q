import cv2
import numpy as np
from sklearn.cluster import KMeans

def create_hsv_mask(image, lower_hsv, upper_hsv):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    return mask

def apply_morphological_operations(mask, itr=1, num_dilations=2, num_erosions=2):
    for _ in range(itr):
        mask = cv2.dilate(mask, None, iterations=num_dilations)
        mask = cv2.erode(mask, None, iterations=num_erosions)
        mask = cv2.erode(mask, None, iterations=num_erosions)
        mask = cv2.dilate(mask, None, iterations=num_dilations)
    return mask

def kmeans_segmentation(image, k=2):
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, center = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    segmented_image = center[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image

# Usage
def solution(image_path):
    ######################################################################
    ######################################################################
    image = cv2.imread(image_path)
    lower_hsv = np.array([0, 100, 100])  
    upper_hsv = np.array([30, 255, 255])
    hsv_mask = create_hsv_mask(image, lower_hsv, upper_hsv)
    hsv_mask_processed = apply_morphological_operations(hsv_mask, itr=10, num_dilations=2, num_erosions=2)
    segmented_image = kmeans_segmentation(image, k=2)
    _, lava_mask = cv2.threshold(segmented_image, 1, 255, cv2.THRESH_BINARY)
    hsv_mask_processed = cv2.resize(hsv_mask_processed, (lava_mask.shape[1], lava_mask.shape[0]))
    final_lava_mask = cv2.bitwise_and(lava_mask, lava_mask, mask=hsv_mask_processed)
    return final_lava_mask
    ######################################################################  
