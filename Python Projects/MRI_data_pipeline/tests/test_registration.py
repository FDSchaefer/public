"""
Unit tests for registration.py functions
"""
import unittest
import numpy as np
import sys
sys.path.insert(0, '/mnt/project/src')

from utils import ImageData
from registration import (
    numpy_to_sitk, sitk_to_numpy, skull_strip, load_atlas,
    register_to_atlas, apply_transform_to_mask, atlas_based_skull_strip
)
from pathlib import Path

# Paths for test data
ATLAS_DIR = Path('/home/fds/Documents/github/omni8task/MNI_atlas')
SAMPLE_DATA_DIR = Path('/home/fds/Documents/github/omni8task/data/sample_data')


class TestNumpySitkConversion(unittest.TestCase):
    """Test conversion between NumPy and SimpleITK formats"""
    
    def test_numpy_to_sitk_basic(self):
        """Test basic NumPy to SimpleITK conversion"""
        data = np.random.rand(10, 10, 10).astype(np.float32)
        img = ImageData(data)
        
        sitk_img = numpy_to_sitk(img)
        
        # Check that conversion succeeded
        self.assertIsNotNone(sitk_img)
        self.assertEqual(sitk_img.GetSize(), (10, 10, 10))
    
    def test_sitk_to_numpy_basic(self):
        """Test basic SimpleITK to NumPy conversion"""
        data = np.random.rand(10, 10, 10).astype(np.float32)
        img = ImageData(data)
        
        sitk_img = numpy_to_sitk(img)
        converted_back = sitk_to_numpy(sitk_img, img)
        
        # Check shape is preserved
        self.assertEqual(converted_back.shape, img.shape)
    
    def test_roundtrip_conversion(self):
        """Test that NumPy -> SITK -> NumPy preserves data"""
        data = np.random.rand(10, 10, 10)
        affine = np.eye(4)
        affine[0, 0] = 2.0  # Set spacing
        affine[1, 1] = 2.0
        affine[2, 2] = 2.0
        img = ImageData(data, affine)
        
        sitk_img = numpy_to_sitk(img)
        converted_back = sitk_to_numpy(sitk_img, img)
        
        # Check data is approximately preserved
        np.testing.assert_array_almost_equal(
            data, 
            converted_back.data, 
            decimal=5
        )
    
    def test_preserves_spacing(self):
        """Test that conversion preserves voxel spacing"""
        data = np.random.rand(10, 10, 10)
        affine = np.eye(4)
        affine[0, 0] = 2.5
        affine[1, 1] = 2.5
        affine[2, 2] = 3.0
        img = ImageData(data, affine)
        
        sitk_img = numpy_to_sitk(img)
        spacing = sitk_img.GetSpacing()
        
        # Check spacing is approximately correct
        np.testing.assert_array_almost_equal(
            spacing, 
            [2.5, 2.5, 3.0], 
            decimal=5
        )


class TestSkullStrip(unittest.TestCase):
    """Test skull stripping function"""
    
    def test_basic_skull_stripping(self):
        """Test basic skull stripping with binary mask"""
        # Create test image
        data = np.ones((10, 10, 10)) * 100
        img = ImageData(data)
        
        # Create binary mask (brain in center)
        mask_data = np.zeros((10, 10, 10))
        mask_data[3:7, 3:7, 3:7] = 1
        mask = ImageData(mask_data)
        
        result = skull_strip(img, mask)
        
        # Check that non-brain region is zeroed
        self.assertEqual(result.data[0, 0, 0], 0)
        # Check that brain region is preserved
        self.assertEqual(result.data[5, 5, 5], 100)
    
    def test_skull_strip_preserves_shape(self):
        """Test that skull stripping preserves image shape"""
        data = np.random.rand(15, 20, 25)
        img = ImageData(data)
        
        mask_data = np.ones((15, 20, 25))
        mask = ImageData(mask_data)
        
        result = skull_strip(img, mask)
        
        self.assertEqual(result.shape, img.shape)
    
    def test_skull_strip_with_zero_mask(self):
        """Test skull stripping with all-zero mask"""
        data = np.ones((10, 10, 10)) * 100
        img = ImageData(data)
        
        mask_data = np.zeros((10, 10, 10))
        mask = ImageData(mask_data)
        
        result = skull_strip(img, mask)
        
        # All voxels should be zero
        np.testing.assert_array_equal(result.data, np.zeros((10, 10, 10)))
    
    def test_skull_strip_with_full_mask(self):
        """Test skull stripping with all-one mask"""
        data = np.random.rand(10, 10, 10) * 100
        img = ImageData(data)
        
        mask_data = np.ones((10, 10, 10))
        mask = ImageData(mask_data)
        
        result = skull_strip(img, mask)
        
        # All voxels should be preserved
        np.testing.assert_array_almost_equal(result.data, data)
    
    def test_mismatched_shapes(self):
        """Test that mismatched shapes raise error"""
        data = np.random.rand(10, 10, 10)
        img = ImageData(data)
        
        mask_data = np.ones((15, 15, 15))
        mask = ImageData(mask_data)
        
        with self.assertRaises(ValueError) as context:
            skull_strip(img, mask)
        self.assertIn("doesn't match", str(context.exception))
    
    def test_skull_strip_preserves_affine(self):
        """Test that skull stripping preserves affine"""
        data = np.random.rand(10, 10, 10)
        affine = np.random.rand(4, 4)
        img = ImageData(data, affine)
        
        mask_data = np.ones((10, 10, 10))
        mask = ImageData(mask_data)
        
        result = skull_strip(img, mask)
        
        np.testing.assert_array_equal(result.affine, affine)
    
    def test_fractional_mask(self):
        """Test skull stripping with fractional mask values"""
        data = np.ones((10, 10, 10)) * 100
        img = ImageData(data)
        
        # Mask with fractional values
        mask_data = np.full((10, 10, 10), 0.5)
        mask = ImageData(mask_data)
        
        result = skull_strip(img, mask)
        
        # Result should be scaled by mask
        expected = data * mask_data
        np.testing.assert_array_almost_equal(result.data, expected)


class TestLoadAtlas(unittest.TestCase):
    """Test atlas loading function"""

    def test_load_atlas_success(self):
        """Test successful atlas loading"""
        template, mask = load_atlas(ATLAS_DIR)

        self.assertIsNotNone(template)
        self.assertIsNotNone(mask)
        self.assertEqual(template.shape, mask.shape)

    def test_load_atlas_missing_directory(self):
        """Test error when atlas directory doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            load_atlas(Path('/nonexistent/path'))

    def test_load_atlas_missing_template(self):
        """Test error when template file is missing"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_atlas(Path(tmpdir))
            self.assertIn("No T1 template", str(ctx.exception))


class TestRegisterToAtlas(unittest.TestCase):
    """Test image registration functions"""

    def test_invalid_registration_type(self):
        """Test that invalid registration type raises error"""
        img = ImageData(np.random.rand(10, 10, 10))
        with self.assertRaises(ValueError) as ctx:
            register_to_atlas(img, img, registration_type="invalid")
        self.assertIn("Unsupported registration type", str(ctx.exception))

    def test_rigid_registration_runs(self):
        """Test that rigid registration executes without error"""
        # Create small test images for speed
        data1 = np.random.rand(20, 20, 20).astype(np.float32)
        data2 = np.random.rand(20, 20, 20).astype(np.float32)
        img1 = ImageData(data1)
        img2 = ImageData(data2)

        registered, transform = register_to_atlas(img1, img2, registration_type="rigid")

        self.assertEqual(registered.shape, img2.shape)
        self.assertIsNotNone(transform)

    def test_affine_registration_runs(self):
        """Test that affine registration executes without error"""
        data1 = np.random.rand(20, 20, 20).astype(np.float32)
        data2 = np.random.rand(20, 20, 20).astype(np.float32)
        img1 = ImageData(data1)
        img2 = ImageData(data2)

        registered, transform = register_to_atlas(img1, img2, registration_type="affine")

        self.assertEqual(registered.shape, img2.shape)
        self.assertIsNotNone(transform)


class TestApplyTransformToMask(unittest.TestCase):
    """Test mask transformation function"""

    def test_apply_transform_preserves_binary(self):
        """Test that transformed mask remains binary"""
        # Create images and get a transform
        data1 = np.random.rand(20, 20, 20).astype(np.float32)
        data2 = np.random.rand(20, 20, 20).astype(np.float32)
        img1 = ImageData(data1)
        img2 = ImageData(data2)

        _, transform = register_to_atlas(img1, img2, registration_type="rigid")

        # Create binary mask
        mask_data = np.zeros((20, 20, 20))
        mask_data[5:15, 5:15, 5:15] = 1
        mask = ImageData(mask_data)

        transformed = apply_transform_to_mask(mask, transform, img1)

        # Check result is binary
        unique_vals = np.unique(transformed.data)
        self.assertTrue(all(v in [0, 1] for v in unique_vals))


class TestAtlasBasedSkullStrip(unittest.TestCase):
    """Test complete skull stripping pipeline"""

    def test_original_target_requires_original_data(self):
        """Test error when mask_target='original' but no original_img_data"""
        img = ImageData(np.random.rand(10, 10, 10))
        with self.assertRaises(ValueError) as ctx:
            atlas_based_skull_strip(
                img, ATLAS_DIR, mask_target="original", original_img_data=None
            )
        self.assertIn("original_img_data must be provided", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
