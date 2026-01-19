"""
Unit tests for quality_assessment.py functions
"""
import unittest
import numpy as np
import sys
sys.path.insert(0, '/mnt/project/src')

from utils import ImageData
from quality_assessment import (
    calculate_mask_coverage,
    calculate_brain_volume,
    check_connected_components,
    calculate_edge_density,
    calculate_intensity_statistics
)


class TestMaskCoverage(unittest.TestCase):
    """Test mask coverage calculation"""
    
    def test_full_coverage(self):
        """Test coverage with all non-zero voxels"""
        data = np.ones((10, 10, 10))
        img = ImageData(data)
        
        coverage = calculate_mask_coverage(img)
        
        self.assertAlmostEqual(coverage, 100.0, places=2)
    
    def test_no_coverage(self):
        """Test coverage with all zero voxels"""
        data = np.zeros((10, 10, 10))
        img = ImageData(data)
        
        coverage = calculate_mask_coverage(img)
        
        self.assertAlmostEqual(coverage, 0.0, places=2)
    
    def test_half_coverage(self):
        """Test coverage with half non-zero voxels"""
        data = np.zeros((10, 10, 10))
        data[:5, :, :] = 1
        img = ImageData(data)
        
        coverage = calculate_mask_coverage(img)
        
        self.assertAlmostEqual(coverage, 50.0, places=2)
    
    def test_small_coverage(self):
        """Test coverage with few non-zero voxels"""
        data = np.zeros((10, 10, 10))
        data[5, 5, 5] = 1  # Single voxel
        img = ImageData(data)
        
        coverage = calculate_mask_coverage(img)
        
        expected = (1 / 1000) * 100
        self.assertAlmostEqual(coverage, expected, places=2)


class TestBrainVolume(unittest.TestCase):
    """Test brain volume calculation"""
    
    def test_volume_calculation(self):
        """Test basic volume calculation"""
        # Create image with known voxel size
        data = np.ones((10, 10, 10))
        affine = np.eye(4)
        affine[0, 0] = 1.0  # 1mm spacing
        affine[1, 1] = 1.0
        affine[2, 2] = 1.0
        img = ImageData(data, affine)
        
        volume = calculate_brain_volume(img)
        
        # 10*10*10 voxels * 1mm^3 = 1000 mm^3 = 1 cm^3
        self.assertAlmostEqual(volume, 1.0, places=2)
    
    def test_volume_with_different_spacing(self):
        """Test volume calculation with different voxel spacing"""
        data = np.ones((10, 10, 10))
        affine = np.eye(4)
        affine[0, 0] = 2.0  # 2mm spacing
        affine[1, 1] = 2.0
        affine[2, 2] = 2.0
        img = ImageData(data, affine)
        
        volume = calculate_brain_volume(img)
        
        # 10*10*10 voxels * 8mm^3 = 8000 mm^3 = 8 cm^3
        self.assertAlmostEqual(volume, 8.0, places=2)
    
    def test_partial_brain(self):
        """Test volume calculation with partial brain"""
        data = np.zeros((10, 10, 10))
        data[2:8, 2:8, 2:8] = 1  # 6x6x6 cube
        affine = np.eye(4)
        affine[0, 0] = 1.0
        affine[1, 1] = 1.0
        affine[2, 2] = 1.0
        img = ImageData(data, affine)
        
        volume = calculate_brain_volume(img)
        
        # 6*6*6 = 216 voxels * 1mm^3 = 216 mm^3 = 0.216 cm^3
        self.assertAlmostEqual(volume, 0.216, places=3)


class TestConnectedComponents(unittest.TestCase):
    """Test connected components analysis"""
    
    def test_single_component(self):
        """Test single connected component"""
        data = np.zeros((10, 10, 10))
        data[3:7, 3:7, 3:7] = 1
        img = ImageData(data)
        
        results = check_connected_components(img)
        
        self.assertEqual(results['num_components'], 1)
        self.assertAlmostEqual(results['largest_component_fraction'], 1.0)
    
    def test_two_components(self):
        """Test two separate components"""
        data = np.zeros((20, 20, 20))
        data[2:5, 2:5, 2:5] = 1    # Component 1
        data[15:18, 15:18, 15:18] = 1  # Component 2
        img = ImageData(data)
        
        results = check_connected_components(img)
        
        self.assertEqual(results['num_components'], 2)
        self.assertLess(results['largest_component_fraction'], 1.0)
    
    def test_no_components(self):
        """Test all-zero image"""
        data = np.zeros((10, 10, 10))
        img = ImageData(data)
        
        results = check_connected_components(img)
        
        self.assertEqual(results['num_components'], 0)
        self.assertEqual(results['largest_component_size'], 0)


class TestEdgeDensity(unittest.TestCase):
    """Test edge density calculation"""
    
    def test_smooth_boundary(self):
        """Test edge density on smooth boundary"""
        # Create smooth sphere
        data = np.zeros((20, 20, 20))
        center = 10
        radius = 5
        for i in range(20):
            for j in range(20):
                for k in range(20):
                    dist = np.sqrt((i-center)**2 + (j-center)**2 + (k-center)**2)
                    if dist < radius:
                        data[i, j, k] = 100
        
        img = ImageData(data)
        
        edge_density = calculate_edge_density(img)
        
        # Should have non-zero edge density
        self.assertGreater(edge_density, 0)
    
    def test_sharp_boundary(self):
        """Test edge density on sharp boundary"""
        data = np.zeros((10, 10, 10))
        data[3:7, 3:7, 3:7] = 100
        img = ImageData(data)
        
        edge_density = calculate_edge_density(img)
        
        # Should have higher edge density than smooth
        self.assertGreater(edge_density, 0)
    
    def test_no_brain(self):
        """Test edge density on all-zero image"""
        data = np.zeros((10, 10, 10))
        img = ImageData(data)
        
        edge_density = calculate_edge_density(img)
        
        self.assertEqual(edge_density, 0)


class TestIntensityStatistics(unittest.TestCase):
    """Test intensity statistics calculation"""
    
    def test_basic_statistics(self):
        """Test basic statistical measures"""
        data = np.zeros((10, 10, 10))
        data[2:8, 2:8, 2:8] = np.random.rand(6, 6, 6)
        img = ImageData(data)
        
        stats = calculate_intensity_statistics(img)
        
        # Check all keys are present
        expected_keys = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check basic sanity
        self.assertGreaterEqual(stats['mean'], 0)
        self.assertGreaterEqual(stats['std'], 0)
        self.assertLessEqual(stats['min'], stats['max'])
        self.assertLessEqual(stats['q25'], stats['median'])
        self.assertLessEqual(stats['median'], stats['q75'])
    
    def test_constant_intensity(self):
        """Test statistics with constant intensity"""
        data = np.zeros((10, 10, 10))
        data[2:8, 2:8, 2:8] = 5.0
        img = ImageData(data)
        
        stats = calculate_intensity_statistics(img)
        
        self.assertAlmostEqual(stats['mean'], 5.0)
        self.assertAlmostEqual(stats['std'], 0.0)
        self.assertAlmostEqual(stats['min'], 5.0)
        self.assertAlmostEqual(stats['max'], 5.0)
    
    def test_empty_brain(self):
        """Test statistics with no brain voxels"""
        data = np.zeros((10, 10, 10))
        img = ImageData(data)
        
        stats = calculate_intensity_statistics(img)
        
        # All stats should be 0 for empty brain
        for value in stats.values():
            self.assertEqual(value, 0)


if __name__ == '__main__':
    unittest.main()
