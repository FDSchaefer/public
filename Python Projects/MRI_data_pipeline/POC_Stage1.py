"""
Proof of concept for Stage 1: Loading and preprocessing
"""
import sys
sys.path.insert(0, './src')

import numpy as np
from utils import ImageData, validate_image_data, setup_logging,load_dicom_series ,load_nifti, save_nifti
from preprocessing import normalize_intensity, apply_gaussian_smoothing, preprocess_image
from scrollview import Scroller, ScrollerMulti, ScrollerCheckerboard
from pathlib import Path

# Setup logging
setup_logging("INFO")

PLT = input("Show Plots? (y/n): ").lower() == 'y'
OUTPUT_DIR = Path("./data/sample_data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load sample image data
print("\n1. Loading sample image data...")
INPUT_FILE = "./data/sample_data/test_sample.nii"
img_data = load_nifti(INPUT_FILE)

#INPUT_FILE = "./data/sample_data/dicom_sample"
#img_data = load_dicom_series(INPUT_FILE)

# Validate
print("\n2. Validating image data...")
try:
    validate_image_data(img_data)
    print("   ✓ Validation passed")
except Exception as e:
    print(f"   ✗ Validation failed: {e}")
    sys.exit(1)

if PLT:
    Scroller(img_data.data)
    pause = input("Press Enter to continue after closing the image viewer...")

# Test normalization methods
print("\n3. Testing normalization...")

print("\n   a) Z-score normalization:")
normalized_zscore = normalize_intensity(img_data, method="zscore")
print(f"      Mean: {np.mean(normalized_zscore.data):.6f} (should be ~0)")
print(f"      Std:  {np.std(normalized_zscore.data):.6f} (should be ~1)")
print(f"      Range: [{np.min(normalized_zscore.data):.2f}, {np.max(normalized_zscore.data):.2f}]")

if PLT:
    Scroller(normalized_zscore.data)
    pause = input("Press Enter to continue after closing the image viewer...")

print("\n   b) Min-max normalization:")
normalized_minmax = normalize_intensity(img_data, method="minmax")
print(f"      Min: {np.min(normalized_minmax.data):.6f} (should be 0)")
print(f"      Max: {np.max(normalized_minmax.data):.6f} (should be 1)")

if PLT:
    Scroller(normalized_minmax.data)
    pause = input("Press Enter to continue after closing the image viewer...")

# Test smoothing
print("\n4. Testing Gaussian smoothing...")
for sigma in [0.5, 1.0, 2.0]:
    smoothed = apply_gaussian_smoothing(img_data, sigma=sigma)
    variance_reduction = (np.var(img_data.data) - np.var(smoothed.data)) / np.var(img_data.data)
    print(f"   σ={sigma}: variance reduced by {variance_reduction*100:.1f}%")

if PLT:
    Scroller(smoothed.data)
    pause = input("Press Enter to continue after closing the image viewer...")

# Test full preprocessing pipeline
print("\n5. Testing complete preprocessing pipeline...")
preprocessed = preprocess_image(img_data, normalize_method="zscore", sigma=1.0)
print(f"   Output shape: {preprocessed.shape}")
print(f"   Output range: [{np.min(preprocessed.data):.2f}, {np.max(preprocessed.data):.2f}]")
print(f"   Mean: {np.mean(preprocessed.data):.6f}")
print(f"   Std:  {np.std(preprocessed.data):.6f}")

print("\n6. Save Processed Image...")
save_nifti(preprocessed, OUTPUT_DIR / "nii_sample_preprocessed.nii")

if PLT:
    ScrollerCheckerboard(normalized_zscore.data,preprocessed.data)
    pause = input("Press Enter to continue after closing the image viewer...")
