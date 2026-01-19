"""
Demo script showcasing scrollview.py visualization options.
Compares two versions of the same image with different processing.
"""
import sys
sys.path.insert(0, './src')

import numpy as np
from utils import load_nifti, setup_logging
from preprocessing import preprocess_image, normalize_intensity, apply_gaussian_smoothing
from scrollview import (
    Scroller,
    ScrollerMulti, 
    ScrollerCheckerboard,
    ScrollerDifference,
    ScrollerOverlay
)

# Setup logging
setup_logging("INFO")

# Load the test sample
print("Loading test sample...")
img_data = load_nifti('./data/sample_data/test_sample.nii')

# Create two versions with different preprocessing
print("\nCreating two versions for comparison:")
print("  Version 1: Light smoothing (sigma=0.5)")
print("  Version 2: Heavy smoothing (sigma=2.0)")

# Version 1: Light preprocessing
img1 = preprocess_image(img_data, normalize_method="zscore", sigma=0.5)

# Version 2: Heavy preprocessing  
img2 = preprocess_image(img_data, normalize_method="zscore", sigma=2.0)

print("\n" + "="*60)
print("SCROLLVIEW VISUALIZATION DEMOS")
print("="*60)

# Demo 1: Single image viewer
print("\n1. Scroller - Single image viewer")
print("   Controls: Scroll to navigate slices")
print("   Close window to continue...")
Scroller(img1.data)

# Demo 2: Multi-image side-by-side comparison
print("\n2. ScrollerMulti - Side-by-side comparison")
print("   Controls: Scroll to navigate slices")
print("   Close window to continue...")
ScrollerMulti(
    [img1.data, img2.data],
    names=["Light Smoothing (σ=0.5)", "Heavy Smoothing (σ=2.0)"]
)

# Demo 3: Checkerboard overlay
print("\n3. ScrollerCheckerboard - Checkerboard overlay")
print("   Controls:")
print("     Scroll: Navigate slices")
print("     'c' or 'b': Toggle checkerboard")
print("     '1': Show only image 1")
print("     '2': Show only image 2")
print("     '+'/'-': Adjust checker size")
print("   Close window to continue...")
ScrollerCheckerboard(
    img1.data, 
    img2.data,
    name1="Light Smoothing",
    name2="Heavy Smoothing",
    checker_size=32
)

# Demo 4: Difference map
print("\n4. ScrollerDifference - Difference visualization")
print("   Shows both images plus their difference")
print("   Controls: Scroll to navigate slices")
print("   Close window to continue...")
ScrollerDifference(
    img1.data,
    img2.data, 
    name1="Light Smoothing",
    name2="Heavy Smoothing"
)

# Demo 5: Blended overlay
print("\n5. ScrollerOverlay - Alpha-blended overlay")
print("   Controls:")
print("     Scroll: Navigate slices")
print("     Left/Right arrows: Adjust blend alpha")
print("   Close window to continue...")
ScrollerOverlay(
    img1.data,
    img2.data,
    name1="Light Smoothing", 
    name2="Heavy Smoothing",
    alpha=0.5
)

print("\n" + "="*60)
print("DEMO COMPLETE")
print("="*60)
print("\nAll visualization options demonstrated:")
print("  1. Scroller          - Single image viewer")
print("  2. ScrollerMulti     - Multiple images side-by-side")
print("  3. ScrollerCheckerboard - Interactive checkerboard overlay")
print("  4. ScrollerDifference   - Difference map visualization")
print("  5. ScrollerOverlay      - Alpha-blended overlay")
