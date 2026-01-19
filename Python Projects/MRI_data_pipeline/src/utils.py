"""
Utility functions for loading and validating medical imaging data.
"""
import logging
from pathlib import Path
from typing import Tuple, Union
import numpy as np


logger = logging.getLogger(__name__)


class ImageData:
    """Container for 3D medical imaging data with metadata."""
    
    def __init__(self, data: np.ndarray, affine: np.ndarray = None, header: dict = None):
        self.data = data
        self.affine = affine if affine is not None else np.eye(4)
        self.header = header if header is not None else {}
        
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype


def validate_image_data(img_data: ImageData) -> bool:
    """
    Validate that image data meets basic requirements.
    
    Args:
        img_data: ImageData object to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if img_data.data.ndim != 3:
        raise ValueError(f"Expected 3D image, got {img_data.data.ndim}D")
    
    if img_data.data.size == 0:
        raise ValueError("Image data is empty")
    
    if not np.isfinite(img_data.data).all():
        raise ValueError("Image contains NaN or infinite values")
    
    logger.info(f"Image validation passed: shape={img_data.shape}, dtype={img_data.dtype}")
    return True


def load_nifti(filepath: Union[str, Path]) -> ImageData:
    """
    Load a NIFTI file.
    
    Args:
        filepath: Path to NIFTI file (.nii or .nii.gz)
        
    Returns:
        ImageData object containing the loaded image
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix not in ['.nii', '.gz']:
        raise ValueError(f"Expected .nii or .nii.gz file, got {filepath.suffix}")
    
    try:
        import nibabel as nib
        img = nib.load(str(filepath))
        return ImageData(img.get_fdata(), img.affine, dict(img.header))
        
    except Exception as e:
        logger.error(f"Failed to load NIFTI file: {e}")
        raise


def load_dicom_series(directory: Union[str, Path]) -> ImageData:
    """
    Load a DICOM series from a directory.
    
    Args:
        directory: Path to directory containing DICOM files
        
    Returns:
        ImageData object containing the loaded series
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Expected directory, got file: {directory}")
    
    try:
        import SimpleITK as sitk
        logger.info(f"Loading DICOM series from: {directory}")
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(directory))
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        data = sitk.GetArrayFromImage(image)
        return ImageData(data)
        
    except Exception as e:
        logger.error(f"Failed to load DICOM series: {e}")
        raise


def save_nifti(img_data: ImageData, filepath: Union[str, Path]) -> None:
    """
    Save ImageData to a NIFTI file.

    Args:
        img_data: ImageData object to save
        filepath: Output path for NIFTI file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        import nibabel as nib
        nifti_img = nib.Nifti1Image(img_data.data, img_data.affine)
        nib.save(nifti_img, str(filepath))

        logger.info(f"Saving NIFTI file: {filepath}")

    except Exception as e:
        logger.error(f"Failed to save NIFTI file: {e}")
        raise


def save_dicom_series(
    img_data: ImageData,
    output_dir: Union[str, Path],
    series_description: str = "Processed Series",
    patient_name: str = "Anonymous",
    patient_id: str = "000000",
    study_description: str = "MRI Study"
) -> None:
    """
    Save ImageData to a DICOM series.

    Args:
        img_data: ImageData object to save
        output_dir: Output directory for DICOM series
        series_description: Description for the DICOM series
        patient_name: Patient name (default: Anonymous)
        patient_id: Patient ID (default: 000000)
        study_description: Study description
    """
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from datetime import datetime
    import SimpleITK as sitk

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Saving DICOM series to: {output_dir}")

        # Convert to SimpleITK image for easier DICOM writing
        sitk_image = sitk.GetImageFromArray(img_data.data)

        # Set spacing and origin from affine if available
        if img_data.affine is not None:
            # Extract spacing from affine matrix (diagonal elements)
            spacing = [
                abs(img_data.affine[0, 0]),
                abs(img_data.affine[1, 1]),
                abs(img_data.affine[2, 2])
            ]
            sitk_image.SetSpacing(spacing)

            # Extract origin from affine matrix (last column)
            origin = [
                img_data.affine[0, 3],
                img_data.affine[1, 3],
                img_data.affine[2, 3]
            ]
            sitk_image.SetOrigin(origin)

        # Setup DICOM metadata
        current_time = datetime.now()

        # Create a writer
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        # Generate UIDs - CRITICAL: These must be the same for all slices in the series
        study_instance_uid = pydicom.uid.generate_uid()  # Same for all slices in study
        series_instance_uid = pydicom.uid.generate_uid()  # Same for all slices in series
        frame_of_reference_uid = pydicom.uid.generate_uid()  # Same for all slices

        # Generate UIDs
        series_tag_values = [
            ("0008|0060", "MR"),  # Modality
            ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
            ("0008|103e", series_description),  # Series Description
            ("0010|0010", patient_name),  # Patient Name
            ("0010|0020", patient_id),  # Patient ID
            ("0008|1030", study_description),  # Study Description
            ("0008|0020", current_time.strftime("%Y%m%d")),  # Study Date
            ("0008|0030", current_time.strftime("%H%M%S")),  # Study Time
            ("0008|0021", current_time.strftime("%Y%m%d")),  # Series Date
            ("0008|0031", current_time.strftime("%H%M%S")),  # Series Time
            ("0020|000d", study_instance_uid),  # Study Instance UID
            ("0020|000e", series_instance_uid),  # Series Instance UID
            ("0020|0052", frame_of_reference_uid),  # Frame of Reference UID
        ]

        # Write each slice as a separate DICOM file
        for i in range(sitk_image.GetDepth()):
            # Extract slice
            slice_image = sitk_image[:, :, i]

            # Convert floating-point images to appropriate integer type for DICOM
            pixel_id = slice_image.GetPixelID()
            if pixel_id in [sitk.sitkFloat32, sitk.sitkFloat64]:
                # Get min/max values to determine appropriate type
                stats = sitk.StatisticsImageFilter()
                stats.Execute(slice_image)
                min_val = stats.GetMinimum()
                max_val = stats.GetMaximum()

                # Rescale to 16-bit unsigned integer range
                slice_image = sitk.Cast(sitk.RescaleIntensity(slice_image,
                                                               outputMinimum=0.0,
                                                               outputMaximum=65535.0),
                                        sitk.sitkUInt16)

            # Set slice-specific tags
            sop_instance_uid = pydicom.uid.generate_uid()  # Unique for each slice
            slice_tag_values = series_tag_values + [
                ("0020|0013", str(i + 1)),  # Instance Number
                ("0008|0018", sop_instance_uid),  # SOP Instance UID (unique per slice)
            ]

            # Apply tags
            for tag, value in slice_tag_values:
                slice_image.SetMetaData(tag, value)

            # Write slice
            output_file = output_dir / f"slice_{i:04d}.dcm"
            writer.SetFileName(str(output_file))
            writer.Execute(slice_image)

        logger.info(f"Successfully saved {sitk_image.GetDepth()} DICOM slices")

    except Exception as e:
        logger.error(f"Failed to save DICOM series: {e}")
        raise


def apply_mask(image: ImageData, mask: ImageData) -> ImageData:
    """
    Apply a binary mask to an image (skull stripping).

    Args:
        image: ImageData object containing the image to mask
        mask: ImageData object containing the binary mask

    Returns:
        ImageData object with mask applied
    """
    # Squeeze extra dimensions from mask if present
    mask_data = np.squeeze(mask.data)

    if image.shape != mask_data.shape:
        raise ValueError(f"Image shape {image.shape} does not match mask shape {mask_data.shape}")

    # Binarize mask if not already binary
    binary_mask = (mask_data > 0).astype(image.dtype)
    masked_data = image.data * binary_mask

    logger.info(f"Applied mask to image: {image.shape}")
    return ImageData(masked_data, image.affine, image.header)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
