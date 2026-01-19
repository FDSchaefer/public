#!/usr/bin/env python3
"""
Wrapper script for MRI skull stripping pipeline.
Provides a simplified interface matching the task description format.

Usage:
    python3 pipeline_CLI.py --input subject2_anat.nii.gz --output processed_output.nii.gz
    python3 pipeline_CLI.py --input-dir /path/to/scans --output-dir /path/to/results
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, './src')

def load_default_config(config_path: Path = None) -> dict:
    """Load default configuration from file."""
    if config_path is None:
        # Try common locations
        possible_paths = [
            Path('./config.json'),
            Path('./data/config/config.json'),
            Path('../config.json'),
        ]
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path and config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    
    # Return sensible defaults if no config found
    return {
        'normalize_method': 'zscore',
        'gaussian_sigma': 1.0,
        'registration_type': 'rigid',
        'mask_target': 'processed',
        'atlas_dir': './MNI_atlas',
        'log_level': 'INFO'
    }


def main():
    parser = argparse.ArgumentParser(
        description='MRI Skull Stripping Pipeline - Simplified Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a single file:
    python3 pipeline_CLI.py --input scan.nii.gz --output result.nii.gz
  
  Process a single file with custom smoothing:
    python3 pipeline_CLI.py --input scan.nii.gz --output result.nii.gz --sigma 1.5
  
  Process directory in batch mode:
    python3 pipeline_CLI.py --input-dir ./scans --output-dir ./results
  
  Process directory in watch mode:
    python3 pipeline_CLI.py --input-dir ./scans --output-dir ./results --watch
        """
    )
    
    # Input/Output options
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input', type=Path, 
                          help='Input MRI file (NIFTI or DICOM)')
    io_group.add_argument('--output', type=Path,
                          help='Output file path (for single file processing)')
    io_group.add_argument('--input-dir', type=Path,
                          help='Input directory containing MRI files')
    io_group.add_argument('--output-dir', type=Path,
                          help='Output directory for results')
    
    # Processing parameters
    proc_group = parser.add_argument_group('Processing Parameters')
    proc_group.add_argument('--sigma', '--gaussian-sigma', type=float,
                           dest='gaussian_sigma',
                           help='Gaussian smoothing sigma (default: from config)')
    proc_group.add_argument('--normalize', '--normalize-method', 
                           choices=['zscore', 'minmax'],
                           dest='normalize_method',
                           help='Normalization method (default: from config)')
    proc_group.add_argument('--registration', '--registration-type',
                           choices=['rigid', 'affine'],
                           dest='registration_type',
                           help='Registration type (default: from config)')
    proc_group.add_argument('--mask-target',
                           choices=['processed', 'original'],
                           help='Apply mask to processed or original image (default: from config)')
    
    # Atlas and configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--atlas-dir', type=Path,
                             help='Path to MNI atlas directory (default: from config)')
    config_group.add_argument('--config', type=Path,
                             help='Path to configuration file (default: auto-detect)')
    config_group.add_argument('--log-level', 
                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                             help='Logging level (default: from config)')
    
    # Mode options
    mode_group = parser.add_argument_group('Mode Options')
    mode_group.add_argument('--watch', action='store_true',
                           help='Run in watch mode (monitor directory for new files)')
    
    args = parser.parse_args()
    
    # Validate input arguments
    has_single = args.input is not None
    has_dir = args.input_dir is not None
    
    if not has_single and not has_dir:
        parser.error('Either --input or --input-dir must be specified')
    
    if has_single and has_dir:
        parser.error('Cannot specify both --input and --input-dir')
    
    if has_single and not args.output:
        parser.error('--output must be specified when using --input')
    
    if has_dir and not args.output_dir:
        parser.error('--output-dir must be specified when using --input-dir')
    
    if args.watch and not has_dir:
        parser.error('--watch can only be used with --input-dir')
    
    # Load default configuration
    config = load_default_config(args.config)
    
    # Override with command-line arguments
    if args.gaussian_sigma is not None:
        config['gaussian_sigma'] = args.gaussian_sigma
    if args.normalize_method is not None:
        config['normalize_method'] = args.normalize_method
    if args.registration_type is not None:
        config['registration_type'] = args.registration_type
    if args.mask_target is not None:
        config['mask_target'] = args.mask_target
    if args.atlas_dir is not None:
        config['atlas_dir'] = str(args.atlas_dir)
    if args.log_level is not None:
        config['log_level'] = args.log_level
    
    # Import pipeline module (assumed to be in src/)
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    try:
        from pipeline import process_single_file, run_watch_mode, run_batch_mode
        from utils import setup_logging
    except ImportError as e:
        print(f"Error: Could not import pipeline module: {e}")
        print("Make sure the 'src' directory is in the correct location")
        sys.exit(1)
    
    # Setup logging
    setup_logging(config['log_level'])
    
    # Process based on mode
    if has_single:
        # Single file processing
        print(f"Processing: {args.input} -> {args.output}")
        print(f"Configuration: {config}")
        
        # Create output directory if needed
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        # Process the file
        success = process_single_file(args.input, config, args.output.parent)
        
        if success:
            # Rename to match user's requested output name
            generated_output = args.output.parent / f"{args.input.stem}_skull_stripped.nii.gz"
            if generated_output.exists() and generated_output != args.output:
                generated_output.rename(args.output)
                print(f"Result saved to: {args.output}")
            sys.exit(0)
        else:
            print("Processing failed")
            sys.exit(1)
    
    else:
        # Directory processing
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.watch:
            # Watch mode
            print(f"Starting watch mode:")
            print(f"  Input directory: {args.input_dir}")
            print(f"  Output directory: {args.output_dir}")
            print(f"  Configuration: {config}")
            print("Press Ctrl+C to stop")
            
            # Create temporary config file for pipeline
            temp_config = args.output_dir / '.temp_config.json'
            with open(temp_config, 'w') as f:
                json.dump(config, f, indent=2)
            
            try:
                run_watch_mode(temp_config, args.input_dir, args.output_dir)
            finally:
                if temp_config.exists():
                    temp_config.unlink()
        else:
            # Batch mode
            print(f"Batch processing:")
            print(f"  Input directory: {args.input_dir}")
            print(f"  Output directory: {args.output_dir}")
            print(f"  Configuration: {config}")
            
            # Create temporary config file for pipeline
            temp_config = args.output_dir / '.temp_config.json'
            with open(temp_config, 'w') as f:
                json.dump(config, f, indent=2)
            
            try:
                run_batch_mode(temp_config, args.input_dir, args.output_dir)
            finally:
                if temp_config.exists():
                    temp_config.unlink()


if __name__ == '__main__':
    main()
