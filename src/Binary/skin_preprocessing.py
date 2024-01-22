# -----------------------------------------------------------------------------
# Preprocessing Class
# Author: Xavier Beltran Urbano and Zain Muhammad
# Date Created: 17-11-2023
# -----------------------------------------------------------------------------

# Import libraries
import numpy as np
from network import SegNetwork
import cv2


# Preprocessing Class
class Preprocessing:
    
    def __init__(self,target_size):
        # Export Model
        network = SegNetwork()
        model = network.exportModel()
        model.load_weights("/notebooks/Model_segmentation/Best_weights.h5")
        self.modelSegmentation=model
        self.target_size=target_size

    def extractROI(self,img, threshold=50):
        # image dimensions
        h, w = img.shape[:2]

        # coordinates of the pixels in the diagonal
        y_coords = list(range(0, h))
        x_coords = list(range(0, w))

        # Mean value of the pixels along the diagonal
        diagonal_values = [np.mean(img[i, i, :]) for i in range(min(h, w))]

        # Find the first and last points where the threshold is crossed
        first_cross = next(i for i, value in enumerate(diagonal_values) if value >= threshold)
        last_cross = len(diagonal_values) - next(
            i for i, value in enumerate(reversed(diagonal_values)) if value >= threshold)

        # Set the coordinates to crop the image
        y1 = max(0, first_cross)
        y2 = min(h, last_cross)
        x1 = max(0, first_cross)
        x2 = min(w, last_cross)

        # Crop the image using the calculated coordinates
        img_new = cv2.resize(img[y1:y2, x1:x2, :],self.target_size[:2])

        if img_new.shape[0] == 0 or img_new.shape[1] == 0:
            img_new = img
        return img_new

    def extractROI_batch(self,batch_img):
        roi_img_batch=[]
        for img in batch_img:
            roi_img_batch.append(self.extractROI(img))
        return np.asarray(roi_img_batch)

    
