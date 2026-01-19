"""
Image registration functions for atlas-based skull stripping.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Literal

import SimpleITK as sitk

from utils import ImageData

logger = logging.getLogger(__name__)


def load_atlas(atlas_dir: Path) -> Tuple[ImageData, ImageData]:
    """
    Load MNI152 atlas template and brain mask.
    
    Args:
        atlas_dir: Directory containing atlas files
        
    Returns:
        Tuple of (template ImageData, mask ImageData)
    """
    atlas_dir = Path(atlas_dir)
    
    # Look for T1 template
    template_candidates = [
        "mni_icbm152_t1_tal_nlin_sym_09a.nii",
        "mni_icbm152_t1_tal_nlin_asym_09a.nii"
    ]
    
    template_path = None
    for candidate in template_candidates:
        path = atlas_dir / candidate
        if path.exists():
            template_path = path
            break
    
    if template_path is None:
        raise FileNotFoundError(f"No T1 template found in {atlas_dir}")
    
    logger.info(f"Loading atlas template: {template_path}")
    
    # Load template using nibabel
    import nibabel as nib
    template_img = nib.load(str(template_path))
    template = ImageData(template_img.get_fdata(), template_img.affine, dict(template_img.header))
    
    # Look for brain mask
    mask_candidates = [
        "mni_icbm152_t1_tal_nlin_sym_09a_mask.nii",
        "mni_icbm152_t1_tal_nlin_asym_09a_mask.nii"
    ]
    
    mask_path = None
    for candidate in mask_candidates:
        path = atlas_dir / candidate
        if path.exists():
            mask_path = path
            break
    
    if mask_path is None:
        raise FileNotFoundError(f"No brain mask found in {atlas_dir}")
    
    logger.info(f"Loading atlas mask: {mask_path}")
    
    mask_img = nib.load(str(mask_path))
    mask = ImageData(mask_img.get_fdata(), mask_img.affine, dict(mask_img.header))
    
    return template, mask


def numpy_to_sitk(img_data: ImageData) -> sitk.Image:
    """
    Convert ImageData to SimpleITK Image.
    
    Args:
        img_data: Input ImageData object
        
    Returns:
        SimpleITK Image
    """
    # SimpleITK expects (x, y, z) ordering
    sitk_img = sitk.GetImageFromArray(img_data.data.astype(np.float32))
    
    # Set spacing and origin from affine matrix
    spacing = np.abs(np.diag(img_data.affine[:3, :3]))
    sitk_img.SetSpacing(spacing.tolist())
    
    origin = img_data.affine[:3, 3]
    sitk_img.SetOrigin(origin.tolist())
    
    # Set direction from affine
    direction_matrix = img_data.affine[:3, :3] / spacing[:, np.newaxis]
    sitk_img.SetDirection(direction_matrix.flatten().tolist())
    
    return sitk_img


def sitk_to_numpy(sitk_img: sitk.Image, reference_data: ImageData) -> ImageData:
    """
    Convert SimpleITK Image back to ImageData.
    
    Args:
        sitk_img: SimpleITK Image
        reference_data: Reference ImageData for affine/header info
        
    Returns:
        ImageData object
    """
    data = sitk.GetArrayFromImage(sitk_img).astype(np.float64)
    
    # Reconstruct affine from SimpleITK metadata
    spacing = np.array(sitk_img.GetSpacing())
    origin = np.array(sitk_img.GetOrigin())
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
    
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin
    
    return ImageData(data, affine, reference_data.header)


def register_to_atlas(
    moving_img: ImageData,
    fixed_img: ImageData,
    registration_type: str = "rigid"
) -> Tuple[ImageData, sitk.Transform]:
    """
    Register moving image to fixed image using SimpleITK.
    
    Args:
        moving_img: Image to be registered (subject scan)
        fixed_img: Target image (atlas template)
        registration_type: Type of registration - 'rigid', 'affine'
        
    Returns:
        Tuple of (registered ImageData, transformation)
    """
    logger.info(f"Starting {registration_type} registration")
    
    # Convert to SimpleITK format
    moving_sitk = numpy_to_sitk(moving_img)
    fixed_sitk = numpy_to_sitk(fixed_img)
    
    # Initialize registration method
    registration = sitk.ImageRegistrationMethod()
    
    # Similarity metric
    registration.SetMetricAsMeanSquares()
    
    # Optimizer settings
    registration.SetOptimizerAsGradientDescent(
        learningRate=0.1,
        numberOfIterations=1000,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    
    # Multi-resolution framework
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Interpolator
    registration.SetInterpolator(sitk.sitkLinear)
    
    # Setup initial transform
    if registration_type == "rigid":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk,
            moving_sitk,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif registration_type == "affine":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk,
            moving_sitk,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    else:
        raise ValueError(f"Unsupported registration type: {registration_type}")
    
    registration.SetInitialTransform(initial_transform, inPlace=False)
    
    # Add observer to track progress
    iteration_count = [0]
    
    def iteration_callback():
        iteration_count[0] += 1
        if iteration_count[0] % 10 == 0:
            logger.debug(f"Iteration {iteration_count[0]}: "
                        f"Metric = {registration.GetMetricValue():.4f}")
    
    registration.AddCommand(sitk.sitkIterationEvent, iteration_callback)
    
    # Execute registration
    logger.info("Executing registration...")
    final_transform = registration.Execute(fixed_sitk, moving_sitk)
    
    logger.info(f"Registration complete. Final metric: {registration.GetMetricValue():.4f}")
    logger.info(f"Optimizer stop condition: {registration.GetOptimizerStopConditionDescription()}")
    
    # Apply transform to moving image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    
    registered_sitk = resampler.Execute(moving_sitk)
    
    # Convert back to ImageData
    registered_img = sitk_to_numpy(registered_sitk, fixed_img)
    
    return registered_img, final_transform


def apply_transform_to_mask(
    mask: ImageData,
    transform: sitk.Transform,
    reference_img: ImageData
) -> ImageData:
    """
    Apply transformation to brain mask.
    
    Args:
        mask: Brain mask in atlas space
        transform: Transformation from registration
        reference_img: Reference image for output space
        
    Returns:
        Transformed mask as ImageData
    """
    logger.info("Applying transform to mask")
    
    # Convert mask to SimpleITK
    mask_sitk = numpy_to_sitk(mask)
    reference_sitk = numpy_to_sitk(reference_img)
    
    # Apply transform with nearest neighbor interpolation for binary mask
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform.GetInverse())
    
    transformed_mask_sitk = resampler.Execute(mask_sitk)
    
    # Convert back to ImageData
    transformed_mask = sitk_to_numpy(transformed_mask_sitk, reference_img)
    
    # Ensure binary mask
    transformed_mask.data = (transformed_mask.data > 0.5).astype(np.float64)
    
    return transformed_mask


def skull_strip(img_data: ImageData, mask: ImageData) -> ImageData:
    """
    Apply brain mask to extract brain region (skull stripping).
    
    Args:
        img_data: Input image
        mask: Binary brain mask
        
    Returns:
        Skull-stripped image
    """
    logger.info("Applying brain mask (skull stripping)")
    
    if img_data.shape != mask.shape:
        raise ValueError(f"Image shape {img_data.shape} doesn't match mask shape {mask.shape}")
    
    # Apply mask
    stripped_data = img_data.data * mask.data
    
    # Calculate statistics
    brain_voxels = np.sum(mask.data > 0)
    total_voxels = np.prod(mask.shape)
    brain_percentage = (brain_voxels / total_voxels) * 100
    
    logger.info(f"Brain mask applied: {brain_voxels} voxels ({brain_percentage:.1f}% of volume)")
    
    return ImageData(stripped_data, img_data.affine, img_data.header)


def atlas_based_skull_strip(
    img_data: ImageData,
    atlas_dir: Path,
    registration_type: str = "rigid",
    normalize_method: str = "zscore",
    mask_target: Literal["original", "processed"] = "processed",
    original_img_data: ImageData = None
) -> ImageData:
    """
    Complete atlas-based skull stripping pipeline.

    Args:
        img_data: Input brain scan (should be preprocessed/normalized)
        atlas_dir: Directory containing atlas files
        registration_type: Registration type ('rigid' or 'affine')
        normalize_method: Normalization method applied to input ('zscore' or 'minmax')
        mask_target: Whether to apply mask to 'original' or 'processed' image
        original_img_data: Original unprocessed image (required if mask_target='original')

    Returns:
        Skull-stripped brain image
    """
    logger.info("Starting atlas-based skull stripping")
    logger.info(f"Mask target: {mask_target}")
    
    # Validate inputs
    if mask_target == "original" and original_img_data is None:
        raise ValueError("original_img_data must be provided when mask_target='original'")

    # Load atlas
    template, atlas_mask = load_atlas(atlas_dir)
    logger.info(f"Atlas template shape: {template.shape}")
    logger.info(f"Atlas mask shape: {atlas_mask.shape}")

    # Apply same normalization to atlas template as was applied to input image
    from preprocessing import normalize_intensity
    template = normalize_intensity(template, method=normalize_method)
    logger.info(f"Applied {normalize_method} normalization to atlas template")
    
    # Register input image to atlas
    registered_img, transform = register_to_atlas(
        moving_img=img_data,
        fixed_img=template,
        registration_type=registration_type
    )
    
    # Apply mask in atlas space
    masked_in_atlas = skull_strip(registered_img, atlas_mask)
    
    # Determine which image to apply the mask to
    if mask_target == "original":
        target_img = original_img_data
        logger.info("Applying mask to original (unprocessed) image")
    else:
        target_img = img_data
        logger.info("Applying mask to preprocessed image")
    
    # Transform result back to original space
    inverse_transform = transform.GetInverse()
    
    # Resample back to original space
    moving_sitk = numpy_to_sitk(masked_in_atlas)
    reference_sitk = numpy_to_sitk(target_img)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(inverse_transform)
    
    result_sitk = resampler.Execute(moving_sitk)
    result = sitk_to_numpy(result_sitk, target_img)
    
    logger.info("Atlas-based skull stripping complete")
    
    return result