"""
Image preprocessing functions: normalization and smoothing.
"""
import logging
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Literal

from utils import ImageData

logger = logging.getLogger(__name__)


def normalize_intensity(
    img_data: ImageData, 
    method: Literal["zscore", "minmax"] = "zscore"
) -> ImageData:
    """
    Normalize image intensities.
    
    Args:
        img_data: Input image data
        method: Normalization method - 'zscore' or 'minmax'
        
    Returns:
        Normalized ImageData object
    """
    data = img_data.data.astype(np.float64)
    
    if method == "zscore":
        # Z-score normalization: (x - mean) / std
        mean = np.mean(data)
        std = np.std(data)
        
        if std < 1e-10:
            logger.warning("Standard deviation near zero, skipping normalization")
            normalized = data
        else:
            normalized = (data - mean) / std
            logger.info(f"Z-score normalization: mean={mean:.2f}, std={std:.2f}")
            
    elif method == "minmax":
        # Min-max normalization: (x - min) / (max - min)
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val - min_val < 1e-10:
            logger.warning("Range near zero, skipping normalization")
            normalized = data
        else:
            normalized = (data - min_val) / (max_val - min_val)
            logger.info(f"Min-max normalization: min={min_val:.2f}, max={max_val:.2f}")
            
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return ImageData(normalized, img_data.affine, img_data.header)


def apply_gaussian_smoothing(
    img_data: ImageData, 
    sigma: float = 1.0
) -> ImageData:
    """
    Apply Gaussian smoothing to the image.
    
    Args:
        img_data: Input image data
        sigma: Standard deviation for Gaussian kernel (in voxels)
        
    Returns:
        Smoothed ImageData object
    """
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    
    logger.info(f"Applying Gaussian smoothing with sigma={sigma}")
    
    # Apply 3D Gaussian filter
    smoothed = gaussian_filter(img_data.data, sigma=sigma, mode='nearest')
    
    return ImageData(smoothed, img_data.affine, img_data.header)


def preprocess_image(
    img_data: ImageData,
    normalize_method: Literal["zscore", "minmax"] = "zscore",
    sigma: float = 1.0
) -> ImageData:
    """
    Complete preprocessing pipeline: normalization + smoothing.
    
    Args:
        img_data: Input image data
        normalize_method: Normalization method
        sigma: Gaussian smoothing sigma
        
    Returns:
        Preprocessed ImageData object
    """
    logger.info("Starting preprocessing pipeline")
    
    # Step 1: Normalize
    normalized = normalize_intensity(img_data, method=normalize_method)
    
    # Step 2: Smooth
    smoothed = apply_gaussian_smoothing(normalized, sigma=sigma)
    
    logger.info("Preprocessing complete")
    
    return smoothed
