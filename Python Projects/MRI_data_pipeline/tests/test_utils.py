"""
Unit tests for utils.py functions
"""
import unittest
import numpy as np
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, '/mnt/project/src')

from utils import (
    ImageData, validate_image_data, load_nifti, save_nifti,
    load_dicom_series, save_dicom_series, setup_logging
)

# Paths for test data
SAMPLE_DICOM_DIR = Path('/home/fds/Documents/github/omni8task/data/sample_data/test_sample')


class TestImageData(unittest.TestCase):
    """Test ImageData container class"""
    
    def test_basic_initialization(self):
        """Test basic ImageData creation"""
        data = np.random.rand(10, 10, 10)
        img = ImageData(data)
        
        self.assertEqual(img.shape, (10, 10, 10))
        self.assertTrue(np.array_equal(img.data, data))
        self.assertTrue(np.array_equal(img.affine, np.eye(4)))
        self.assertEqual(img.header, {})
    
    def test_initialization_with_affine_and_header(self):
        """Test ImageData creation with affine and header"""
        data = np.random.rand(10, 10, 10)
        affine = np.random.rand(4, 4)
        header = {'test': 'value'}
        
        img = ImageData(data, affine, header)
        
        self.assertTrue(np.array_equal(img.affine, affine))
        self.assertEqual(img.header, header)
    
    def test_dtype_property(self):
        """Test dtype property"""
        data = np.random.rand(10, 10, 10).astype(np.float32)
        img = ImageData(data)
        
        self.assertEqual(img.dtype, np.float32)


class TestValidateImageData(unittest.TestCase):
    """Test image validation function"""
    
    def test_valid_3d_image(self):
        """Test validation passes for valid 3D image"""
        data = np.random.rand(10, 10, 10)
        img = ImageData(data)
        
        self.assertTrue(validate_image_data(img))
    
    def test_invalid_dimensions(self):
        """Test validation fails for non-3D image"""
        data = np.random.rand(10, 10)  # 2D
        img = ImageData(data)
        
        with self.assertRaises(ValueError) as context:
            validate_image_data(img)
        self.assertIn("Expected 3D image", str(context.exception))
    
    def test_empty_image(self):
        """Test validation fails for empty image"""
        data = np.array([[[]]]).reshape(0, 0, 0)
        img = ImageData(data)
        
        with self.assertRaises(ValueError) as context:
            validate_image_data(img)
        self.assertIn("empty", str(context.exception))
    
    def test_nan_values(self):
        """Test validation fails for NaN values"""
        data = np.random.rand(10, 10, 10)
        data[5, 5, 5] = np.nan
        img = ImageData(data)
        
        with self.assertRaises(ValueError) as context:
            validate_image_data(img)
        self.assertIn("NaN", str(context.exception))
    
    def test_inf_values(self):
        """Test validation fails for infinite values"""
        data = np.random.rand(10, 10, 10)
        data[5, 5, 5] = np.inf
        img = ImageData(data)
        
        with self.assertRaises(ValueError) as context:
            validate_image_data(img)
        self.assertIn("infinite", str(context.exception))


class TestLoadSaveNifti(unittest.TestCase):
    """Test NIFTI loading and saving functions"""
    
    def test_save_and_load_roundtrip(self):
        """Test that save->load preserves data"""
        # Create test data
        original_data = np.random.rand(10, 10, 10)
        original_img = ImageData(original_data)
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.nii"
            save_nifti(original_img, filepath)
            
            # Load it back
            loaded_img = load_nifti(filepath)
            
            # Check data is preserved
            np.testing.assert_array_almost_equal(
                original_img.data, 
                loaded_img.data, 
                decimal=5
            )
            self.assertEqual(original_img.shape, loaded_img.shape)
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error"""
        with self.assertRaises(FileNotFoundError):
            load_nifti("/nonexistent/path/file.nii")
    
    def test_load_invalid_extension(self):
        """Test loading file with invalid extension raises error"""
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            with self.assertRaises(ValueError) as context:
                load_nifti(tmp.name)
            self.assertIn("Expected .nii or .nii.gz", str(context.exception))
    
    def test_save_creates_directory(self):
        """Test that save_nifti creates parent directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "dir" / "test.nii"
            data = np.random.rand(10, 10, 10)
            img = ImageData(data)
            
            save_nifti(img, nested_path)
            
            self.assertTrue(nested_path.exists())


class TestLoadDicomSeries(unittest.TestCase):
    """Test DICOM series loading function"""

    def test_load_dicom_series_success(self):
        """Test successful DICOM series loading from sample data"""
        img = load_dicom_series(SAMPLE_DICOM_DIR)

        self.assertIsNotNone(img)
        self.assertEqual(len(img.shape), 3)
        self.assertGreater(img.data.size, 0)

    def test_load_dicom_returns_imagedata(self):
        """Test that load_dicom_series returns ImageData object"""
        img = load_dicom_series(SAMPLE_DICOM_DIR)

        self.assertIsInstance(img, ImageData)
        self.assertIsInstance(img.data, np.ndarray)

    def test_load_dicom_nonexistent_directory(self):
        """Test error when directory doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            load_dicom_series(Path('/nonexistent/directory'))

    def test_load_dicom_file_instead_of_dir(self):
        """Test error when path is file instead of directory"""
        with tempfile.NamedTemporaryFile() as tmp:
            with self.assertRaises(ValueError) as ctx:
                load_dicom_series(tmp.name)
            self.assertIn("Expected directory", str(ctx.exception))


class TestSaveDicomSeries(unittest.TestCase):
    """Test DICOM series saving function"""

    def test_save_dicom_series_creates_files(self):
        """Test that save_dicom_series creates DICOM files"""
        data = np.random.rand(10, 64, 64).astype(np.float32)
        img = ImageData(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output_dicom"
            save_dicom_series(img, output_dir)

            # Check files were created
            dcm_files = list(output_dir.glob("*.dcm"))
            self.assertEqual(len(dcm_files), 10)

    def test_save_dicom_roundtrip(self):
        """Test save->load roundtrip preserves data shape"""
        data = np.random.rand(5, 32, 32).astype(np.float32)
        img = ImageData(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output_dicom"
            save_dicom_series(img, output_dir)

            # Load it back
            loaded = load_dicom_series(output_dir)
            self.assertEqual(loaded.shape, img.shape)

    def test_save_dicom_with_custom_metadata(self):
        """Test saving with custom patient metadata"""
        data = np.random.rand(5, 32, 32).astype(np.float32)
        img = ImageData(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output_dicom"
            save_dicom_series(
                img, output_dir,
                series_description="Test Series",
                patient_name="Test Patient",
                patient_id="12345"
            )

            dcm_files = list(output_dir.glob("*.dcm"))
            self.assertGreater(len(dcm_files), 0)


class TestSetupLogging(unittest.TestCase):
    """Test logging setup function"""

    def test_setup_logging_info(self):
        """Test logging setup with INFO level"""
        setup_logging("INFO")
        # No exception means success

    def test_setup_logging_debug(self):
        """Test logging setup with DEBUG level"""
        setup_logging("DEBUG")

    def test_setup_logging_case_insensitive(self):
        """Test logging setup is case insensitive"""
        setup_logging("info")
        setup_logging("Info")


if __name__ == '__main__':
    unittest.main()
