from extract_image_feature import get_vector
import cv2
import numpy as np
import os

img_folder = "digital_art/"
feature_folder = "features_database/"

for img in os.listdir('digital_art'):
    features = get_vector(img_folder + img)
    np.save(feature_folder + img.split(".")[0] + ".npy", features)
    
