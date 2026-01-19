"""
Unit tests for preprocessing.py functions
"""
import unittest
import numpy as np
import sys
sys.path.insert(0, '/mnt/project/src')

from utils import ImageData
from preprocessing import normalize_intensity, apply_gaussian_smoothing, preprocess_image


class TestNormalizeIntensity(unittest.TestCase):
    """Test intensity normalization functions"""
    
    def test_zscore_normalization(self):
        """Test z-score normalization produces mean=0, std=1"""
        data = np.random.rand(10, 10, 10) * 100 + 50
        img = ImageData(data)
        
        normalized = normalize_intensity(img, method="zscore")
        
        # Check mean is approximately 0
        self.assertAlmostEqual(np.mean(normalized.data), 0.0, places=6)
        # Check std is approximately 1
        self.assertAlmostEqual(np.std(normalized.data), 1.0, places=6)
    
    def test_minmax_normalization(self):
        """Test min-max normalization produces range [0, 1]"""
        data = np.random.rand(10, 10, 10) * 100 + 50
        img = ImageData(data)
        
        normalized = normalize_intensity(img, method="minmax")
        
        # Check min is 0
        self.assertAlmostEqual(np.min(normalized.data), 0.0, places=6)
        # Check max is 1
        self.assertAlmostEqual(np.max(normalized.data), 1.0, places=6)
    
    def test_constant_image_zscore(self):
        """Test z-score normalization with constant image"""
        data = np.ones((10, 10, 10)) * 5.0
        img = ImageData(data)
        
        # Should handle zero std gracefully
        normalized = normalize_intensity(img, method="zscore")
        
        # Should return unchanged data
        np.testing.assert_array_almost_equal(normalized.data, data)
    
    def test_constant_image_minmax(self):
        """Test min-max normalization with constant image"""
        data = np.ones((10, 10, 10)) * 5.0
        img = ImageData(data)
        
        # Should handle zero range gracefully
        normalized = normalize_intensity(img, method="minmax")
        
        # Should return unchanged data
        np.testing.assert_array_almost_equal(normalized.data, data)
    
    def test_invalid_method(self):
        """Test invalid normalization method raises error"""
        data = np.random.rand(10, 10, 10)
        img = ImageData(data)
        
        with self.assertRaises(ValueError) as context:
            normalize_intensity(img, method="invalid")
        self.assertIn("Unknown normalization method", str(context.exception))
    
    def test_preserves_affine_and_header(self):
        """Test normalization preserves affine and header"""
        data = np.random.rand(10, 10, 10)
        affine = np.random.rand(4, 4)
        header = {'test': 'value'}
        img = ImageData(data, affine, header)
        
        normalized = normalize_intensity(img, method="zscore")
        
        np.testing.assert_array_equal(normalized.affine, affine)
        self.assertEqual(normalized.header, header)


class TestGaussianSmoothing(unittest.TestCase):
    """Test Gaussian smoothing function"""
    
    def test_smoothing_reduces_variance(self):
        """Test that smoothing reduces variance"""
        # Create noisy data
        np.random.seed(42)
        data = np.random.rand(20, 20, 20)
        img = ImageData(data)
        
        original_var = np.var(data)
        
        smoothed = apply_gaussian_smoothing(img, sigma=1.0)
        smoothed_var = np.var(smoothed.data)
        
        # Smoothing should reduce variance
        self.assertLess(smoothed_var, original_var)
    
    def test_larger_sigma_more_smoothing(self):
        """Test that larger sigma produces more smoothing"""
        np.random.seed(42)
        data = np.random.rand(20, 20, 20)
        img = ImageData(data)
        
        smoothed_1 = apply_gaussian_smoothing(img, sigma=1.0)
        smoothed_2 = apply_gaussian_smoothing(img, sigma=2.0)
        
        var_1 = np.var(smoothed_1.data)
        var_2 = np.var(smoothed_2.data)
        
        # Larger sigma should reduce variance more
        self.assertLess(var_2, var_1)
    
    def test_invalid_sigma(self):
        """Test that negative or zero sigma raises error"""
        data = np.random.rand(10, 10, 10)
        img = ImageData(data)
        
        with self.assertRaises(ValueError) as context:
            apply_gaussian_smoothing(img, sigma=0)
        self.assertIn("Sigma must be positive", str(context.exception))
        
        with self.assertRaises(ValueError):
            apply_gaussian_smoothing(img, sigma=-1.0)
    
    def test_preserves_shape(self):
        """Test that smoothing preserves image shape"""
        data = np.random.rand(10, 15, 20)
        img = ImageData(data)
        
        smoothed = apply_gaussian_smoothing(img, sigma=1.0)
        
        self.assertEqual(smoothed.shape, img.shape)
    
    def test_preserves_affine_and_header(self):
        """Test smoothing preserves affine and header"""
        data = np.random.rand(10, 10, 10)
        affine = np.random.rand(4, 4)
        header = {'test': 'value'}
        img = ImageData(data, affine, header)
        
        smoothed = apply_gaussian_smoothing(img, sigma=1.0)
        
        np.testing.assert_array_equal(smoothed.affine, affine)
        self.assertEqual(smoothed.header, header)


class TestPreprocessImage(unittest.TestCase):
    """Test complete preprocessing pipeline"""
    
    def test_preprocessing_pipeline(self):
        """Test that preprocessing applies both normalization and smoothing"""
        np.random.seed(42)
        data = np.random.rand(20, 20, 20) * 100 + 50
        img = ImageData(data)
        
        preprocessed = preprocess_image(img, normalize_method="zscore", sigma=1.0)
        
        # Should have normalized (approximately mean=0, std=1)
        mean = np.mean(preprocessed.data)
        std = np.std(preprocessed.data)
        
        # Due to smoothing, won't be exactly 0 and 1, but should be close
        self.assertLess(abs(mean), 0.1)
        self.assertLess(std, 1.2)
    
    def test_preprocessing_with_minmax(self):
        """Test preprocessing with min-max normalization"""
        data = np.random.rand(20, 20, 20) * 100 + 50
        img = ImageData(data)
        
        preprocessed = preprocess_image(img, normalize_method="minmax", sigma=1.0)
        
        # Should be approximately in [0, 1] range
        self.assertGreaterEqual(np.min(preprocessed.data), -0.1)
        self.assertLessEqual(np.max(preprocessed.data), 1.1)
    
    def test_preserves_shape(self):
        """Test preprocessing preserves image shape"""
        data = np.random.rand(10, 15, 20)
        img = ImageData(data)
        
        preprocessed = preprocess_image(img)
        
        self.assertEqual(preprocessed.shape, img.shape)


if __name__ == '__main__':
    unittest.main()
