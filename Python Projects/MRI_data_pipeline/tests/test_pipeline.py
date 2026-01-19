"""
Unit tests for pipeline.py functions
"""
import unittest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import numpy as np
import sys
sys.path.insert(0, '/mnt/project/src')

from utils import ImageData, save_nifti
from pipeline import (
    process_single_file,
    is_valid_nifti,
    is_already_processed,
    MRIFileHandler
)


class TestIsValidNifti(unittest.TestCase):
    """Test NIFTI file validation"""
    
    def test_valid_nii_extension(self):
        """Test .nii files are valid"""
        filepath = Path("/path/to/image.nii")
        self.assertTrue(is_valid_nifti(filepath))
    
    def test_valid_nii_gz_extension(self):
        """Test .nii.gz files are valid"""
        filepath = Path("/path/to/image.nii.gz")
        self.assertTrue(is_valid_nifti(filepath))
    
    def test_invalid_extension(self):
        """Test non-NIFTI files are invalid"""
        invalid_files = [
            Path("/path/to/image.txt"),
            Path("/path/to/image.jpg"),
            Path("/path/to/image.dcm"),
            Path("/path/to/image.npy"),
        ]
        for filepath in invalid_files:
            self.assertFalse(is_valid_nifti(filepath))
    
    def test_no_extension(self):
        """Test files without extension are invalid"""
        filepath = Path("/path/to/image")
        self.assertFalse(is_valid_nifti(filepath))


class TestIsAlreadyProcessed(unittest.TestCase):
    """Test processed file detection"""
    
    def test_not_processed_no_marker(self):
        """Test file is not processed when no marker exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.nii"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            self.assertFalse(is_already_processed(input_file, output_dir))
    
    def test_processed_with_marker(self):
        """Test file is detected as processed when marker exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.nii"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # Create processed marker
            marker = output_dir / f".{input_file.name}.processed"
            marker.touch()
            
            self.assertTrue(is_already_processed(input_file, output_dir))
    
    def test_processed_with_error_marker(self):
        """Test file is detected as processed when error marker exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.nii"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # Create error marker
            error_marker = output_dir / f".{input_file.name}.error"
            error_marker.touch()
            
            self.assertTrue(is_already_processed(input_file, output_dir))


class TestProcessSingleFile(unittest.TestCase):
    """Test single file processing function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp_dir.name) / "input"
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create test config
        self.config = {
            'normalize_method': 'zscore',
            'gaussian_sigma': 1.0,
            'registration_type': 'rigid',
            'mask_target': 'processed',
            'atlas_dir': '/fake/atlas',
            'log_level': 'INFO'
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.temp_dir.cleanup()
    
    @patch('pipeline.load_nifti')
    @patch('pipeline.preprocess_image')
    @patch('pipeline.atlas_based_skull_strip')
    @patch('pipeline.save_nifti')
    @patch('pipeline.assess_quality')
    @patch('pipeline.save_quality_report_json')
    def test_successful_processing(self, mock_save_report, mock_assess, 
                                   mock_save, mock_strip, mock_preprocess, mock_load):
        """Test successful file processing"""
        # Setup mocks
        input_file = self.input_dir / "test.nii"
        input_file.touch()
        
        mock_img = MagicMock()
        mock_img.shape = (10, 10, 10)
        mock_load.return_value = mock_img
        mock_preprocess.return_value = mock_img
        mock_strip.return_value = mock_img
        mock_assess.return_value = {'overall_pass': True, 'passed_checks': 5, 'total_checks': 5}
        
        # Execute
        result = process_single_file(input_file, self.config, self.output_dir)
        
        # Assert
        self.assertTrue(result)
        mock_load.assert_called_once_with(input_file)
        mock_preprocess.assert_called_once()
        mock_strip.assert_called_once()
        mock_save.assert_called_once()
        mock_assess.assert_called_once()
        mock_save_report.assert_called_once()
        
        # Check marker file created
        marker = self.output_dir / f".{input_file.name}.processed"
        self.assertTrue(marker.exists())
    
    @patch('pipeline.load_nifti')
    def test_processing_with_load_error(self, mock_load):
        """Test processing handles load errors"""
        input_file = self.input_dir / "test.nii"
        input_file.touch()
        
        # Setup mock to raise error
        mock_load.side_effect = Exception("Failed to load")
        
        # Execute
        result = process_single_file(input_file, self.config, self.output_dir)
        
        # Assert
        self.assertFalse(result)
        
        # Check error marker created
        error_marker = self.output_dir / f".{input_file.name}.error"
        self.assertTrue(error_marker.exists())
    
    @patch('pipeline.load_nifti')
    @patch('pipeline.preprocess_image')
    @patch('pipeline.atlas_based_skull_strip')
    def test_processing_with_registration_error(self, mock_strip, mock_preprocess, mock_load):
        """Test processing handles registration errors"""
        input_file = self.input_dir / "test.nii"
        input_file.touch()
        
        mock_img = MagicMock()
        mock_load.return_value = mock_img
        mock_preprocess.return_value = mock_img
        mock_strip.side_effect = Exception("Registration failed")
        
        # Execute
        result = process_single_file(input_file, self.config, self.output_dir)
        
        # Assert
        self.assertFalse(result)
        error_marker = self.output_dir / f".{input_file.name}.error"
        self.assertTrue(error_marker.exists())
    
    @patch('pipeline.load_nifti')
    @patch('pipeline.preprocess_image')
    @patch('pipeline.atlas_based_skull_strip')
    @patch('pipeline.save_nifti')
    @patch('pipeline.assess_quality')
    @patch('pipeline.save_quality_report_json')
    def test_processing_with_mask_target_original(self, mock_save_report, mock_assess,
                                                   mock_save, mock_strip, 
                                                   mock_preprocess, mock_load):
        """Test processing with mask_target='original'"""
        input_file = self.input_dir / "test.nii"
        input_file.touch()
        
        self.config['mask_target'] = 'original'
        
        mock_img = MagicMock()
        mock_load.return_value = mock_img
        mock_preprocess.return_value = mock_img
        mock_strip.return_value = mock_img
        mock_assess.return_value = {'overall_pass': True, 'passed_checks': 5, 'total_checks': 5}
        
        # Execute
        result = process_single_file(input_file, self.config, self.output_dir)
        
        # Assert
        self.assertTrue(result)
        
        # Check that atlas_based_skull_strip was called with original_img_data
        call_kwargs = mock_strip.call_args[1]
        self.assertIn('original_img_data', call_kwargs)
        self.assertEqual(call_kwargs['mask_target'], 'original')


class TestMRIFileHandler(unittest.TestCase):
    """Test MRI file handler for watch mode"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.output_dir.mkdir()
        
        self.config = {
            'normalize_method': 'zscore',
            'gaussian_sigma': 1.0,
            'registration_type': 'rigid',
            'mask_target': 'processed',
            'atlas_dir': '/fake/atlas',
            'log_level': 'INFO'
        }
        
        self.handler = MRIFileHandler(self.config, self.output_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.temp_dir.cleanup()
    
    def test_ignores_directory_events(self):
        """Test handler ignores directory creation"""
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/path/to/directory"
        
        # Should not raise any errors
        self.handler.on_created(event)
    
    def test_ignores_non_nifti_files(self):
        """Test handler ignores non-NIFTI files"""
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/path/to/file.txt"
        
        # Should not raise any errors
        self.handler.on_created(event)
    
    @patch('pipeline.process_single_file')
    def test_processes_nifti_file(self, mock_process):
        """Test handler processes NIFTI files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "test.nii"
            input_file.touch()
            
            # Wait for file to be "written"
            time.sleep(0.1)
            
            event = MagicMock()
            event.is_directory = False
            event.src_path = str(input_file)
            
            mock_process.return_value = True
            
            # Execute
            self.handler.on_created(event)
            
            # Assert - give it a moment to process
            time.sleep(0.5)
            mock_process.assert_called_once()
    
    @patch('pipeline.is_already_processed')
    @patch('pipeline.process_single_file')
    def test_skips_already_processed_files(self, mock_process, mock_is_processed):
        """Test handler skips already processed files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "test.nii"
            input_file.touch()
            
            event = MagicMock()
            event.is_directory = False
            event.src_path = str(input_file)
            
            mock_is_processed.return_value = True
            
            # Execute
            self.handler.on_created(event)
            
            # Assert
            time.sleep(0.5)
            mock_process.assert_not_called()
    
    def test_prevents_duplicate_processing(self):
        """Test handler prevents processing same file twice"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "test.nii"
            
            # Add to processing set
            self.handler.processing.add(input_file)
            
            event = MagicMock()
            event.is_directory = False
            event.src_path = str(input_file)
            
            # Should not process
            with patch('pipeline.process_single_file') as mock_process:
                self.handler.on_created(event)
                mock_process.assert_not_called()

if __name__ == '__main__':
    unittest.main()
