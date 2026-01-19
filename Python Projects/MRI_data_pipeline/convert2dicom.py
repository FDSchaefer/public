"""
Proof of concept for Stage 1: Loading and preprocessing
"""
import sys
sys.path.insert(0, './src')

import numpy as np
from utils import ImageData, load_nifti, save_dicom_series, load_dicom_series
from scrollview import ScrollerMulti

INPUT_FILE = "./data/sample_data/test_sample.nii"
OUTPUT_DIR = "./data/sample_data/test_sample"
img_data = load_nifti(INPUT_FILE)
print(img_data.shape)
save_dicom_series(img_data,OUTPUT_DIR)

img_data_conv = load_dicom_series(OUTPUT_DIR)
print(img_data_conv.shape)
ScrollerMulti([img_data.data,img_data_conv.data])