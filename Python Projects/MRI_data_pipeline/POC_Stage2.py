"""
Proof of concept for Stage 2: Registration and Skull Stripping
"""
import sys
sys.path.insert(0, './src')

import numpy as np
from pathlib import Path
from utils import ImageData, validate_image_data, setup_logging, load_nifti, save_nifti
from preprocessing import preprocess_image
from registration import (
    load_atlas, 
    register_to_atlas, 
    atlas_based_skull_strip,
    skull_strip,
    apply_transform_to_mask
)
from scrollview import Scroller, ScrollerMulti, ScrollerCheckerboard, ScrollerOverlay
from quality_assessment import assess_quality, print_quality_report

# Setup logging
setup_logging("INFO")

PLT = input("Show Plots? (y/n): ").lower() == 'y'

# Configuration
INPUT_FILE = "./data/sample_data/test_sample.nii"
GROUND_TRUTH_FILE = "./data/sample_data/test_sample_manual_strip.nii"
ATLAS_DIR = Path("./MNI_atlas")
OUTPUT_DIR = Path("./data/sample_data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

normalize_method = 'minmax'
registration_type = 'rigid'

print("\n" + "="*60)
print("STAGE 2: REGISTRATION AND SKULL STRIPPING")
print("="*60)

# Step 1: Load input image
print("\n1. Loading input image...")
img_data = load_nifti(INPUT_FILE)
validate_image_data(img_data)
print(f"   Input shape: {img_data.shape}")
print(f"   Input range: [{np.min(img_data.data):.2f}, {np.max(img_data.data):.2f}]")

if PLT:
    Scroller(img_data.data)
    input("Press Enter to continue after closing the image viewer...")

# Step 2: Preprocess
print("\n2. Preprocessing input image...")
preprocessed = preprocess_image(img_data, normalize_method=normalize_method, sigma=1.0)
print(f"   Preprocessed shape: {preprocessed.shape}")
print(f"   Preprocessed range: [{np.min(preprocessed.data):.2f}, {np.max(preprocessed.data):.2f}]")

if PLT:
    Scroller(preprocessed.data)
    input("Press Enter to continue after closing the image viewer...")

# Step 3: Load atlas
print("\n3. Loading MNI152 atlas...")
try:
    template, atlas_mask = load_atlas(ATLAS_DIR)
    print(f"   Template shape: {template.shape}")
    print(f"   Template range: [{np.min(template.data):.2f}, {np.max(template.data):.2f}]")
    print(f"   Mask shape: {atlas_mask.shape}")
    print(f"   Mask coverage: {np.sum(atlas_mask.data > 0) / np.prod(atlas_mask.shape) * 100:.1f}%")
    
    # Apply same normalization to atlas template as was applied to input image
    from preprocessing import normalize_intensity
    template = normalize_intensity(template, method=normalize_method)
    
    if PLT:
        ScrollerMulti(
            [template.data, atlas_mask.data],
            ["Atlas Template", "Atlas Mask"]
        )
        input("Press Enter to continue after closing the image viewer...")
        
except FileNotFoundError as e:
    print(f"   ERROR: {e}")
    print("   Please ensure MNI152 atlas is downloaded to ./MNI_atlas/")
    sys.exit(1)

# Step 4: Registration
print("\n4. Registering input to atlas...")
print("   This may take a few minutes...")
registered_img, transform = register_to_atlas(
    moving_img=preprocessed,
    fixed_img=template,
    registration_type=registration_type
)
print(f"   Registered shape: {registered_img.shape}")

if PLT:
    ScrollerCheckerboard(
        template.data, registered_img.data,
        2,
        ["Atlas Template", "Registered Input"]
    )
    input("Press Enter to continue after closing the image viewer...")

# Step 5: Apply mask
print("\n5. Applying brain mask...")
skull_stripped_in_atlas = skull_strip(registered_img, atlas_mask)
print(f"   Skull-stripped shape: {skull_stripped_in_atlas.shape}")

if PLT:
    ScrollerMulti(
        [registered_img.data, skull_stripped_in_atlas.data],
        ["Before Mask", "After Mask"]
    )
    input("Press Enter to continue after closing the image viewer...")

# Step 6: Transform mask back to original space
print("\n6. Transforming mask to original space...")
mask_in_original_space = apply_transform_to_mask(
    mask=atlas_mask,
    transform=transform,
    reference_img=preprocessed
)
print(f"   Transformed mask shape: {mask_in_original_space.shape}")

# Step 7: Apply mask in original space
print("\n7. Applying mask in original space...")
final_result = skull_strip(preprocessed, mask_in_original_space)
print(f"   Final result shape: {final_result.shape}")
print(f"   Final result range: [{np.min(final_result.data):.2f}, {np.max(final_result.data):.2f}]")

# Calculate quality metrics
non_zero_voxels = np.sum(final_result.data != 0)
total_voxels = np.prod(final_result.shape)
print(f"   Brain coverage: {non_zero_voxels / total_voxels * 100:.1f}%")

if PLT:
    ScrollerMulti(
        [preprocessed.data, mask_in_original_space.data, final_result.data],
        ["Preprocessed", "Mask", "Skull-Stripped"]
    )
    input("Press Enter to continue after closing the image viewer...")

# Step 8: Save results
print("\n8. Saving results...")
save_nifti(registered_img, OUTPUT_DIR / "registered_to_atlas.nii")
save_nifti(skull_stripped_in_atlas, OUTPUT_DIR / "skull_stripped_in_atlas.nii")
save_nifti(mask_in_original_space, OUTPUT_DIR / "mask_in_original_space.nii")
save_nifti(final_result, OUTPUT_DIR / "skull_stripped_final.nii")
print("   All results saved")

# Alternative: Complete pipeline in one call
print("\n9. Testing complete pipeline...")
complete_result = atlas_based_skull_strip(
    img_data=preprocessed,
    atlas_dir=ATLAS_DIR,
    registration_type=registration_type,
    normalize_method = normalize_method,
    mask_target='original',
    original_img_data=img_data
)
save_nifti(complete_result, OUTPUT_DIR / "skull_stripped_pipeline.nii")
print("   Pipeline result saved")

if PLT:
    ScrollerMulti(
        [img_data.data,preprocessed.data, complete_result.data],
        ["Original","Preprocessed", "Pipeline Result"]
    )
    input("Press Enter to continue after closing the image viewer...")

# Step 10: Quality assessment against ground truth
print("\n10. Quality assessment against manual segmentation...")
ground_truth = load_nifti(GROUND_TRUTH_FILE)
quality_results = assess_quality(final_result, ground_truth_mask=ground_truth)
print_quality_report(quality_results, filename=INPUT_FILE)
