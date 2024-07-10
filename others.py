import numpy as np
import cv2

def is_within_bounds(cx, cy, point1, point2):
    x_min = min(point1[0], point2[0])
    x_max = max(point1[0], point2[0])
    y_min = min(point1[1], point2[1])
    y_max = max(point1[1], point2[1])
    
    return x_min <= cx <= x_max and y_min <= cy <= y_max


def thresholding(image):
    binarized_frame = np.zeros_like(image, dtype=np.uint8)
    threshold_value = 30
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > threshold_value:
                binarized_frame[i, j] = 255
            else:
                binarized_frame[i, j] = 0
    return binarized_frame

def manual_median(frames):
    num_frames = len(frames)
    height, width, channels = frames[0].shape
    median_frame = np.zeros((height, width, channels), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                pixel_values = [frames[f][i, j, k] for f in range(num_frames)]
                median_frame[i, j, k] = sorted(pixel_values)[num_frames // 2]
                
    return median_frame

def dilate(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    dilated_image = np.zeros_like(image)
    
    for i in range(pad_height, padded_image.shape[0] - pad_height):
        for j in range(pad_width, padded_image.shape[1] - pad_width):
            region = padded_image[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1]
            dilated_image[i-pad_height, j-pad_width] = np.max(region * kernel)
    
    return dilated_image

def erode(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    eroded_image = np.zeros_like(image)
    
    for i in range(pad_height, padded_image.shape[0] - pad_height):
        for j in range(pad_width, padded_image.shape[1] - pad_width):
            region = padded_image[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1]
            eroded_image[i-pad_height, j-pad_width] = np.min(region * kernel)
    
    return eroded_image

def morphology_open(image, kernel):
    eroded = erode(image, kernel)
    opened = dilate(eroded, kernel)
    return opened
