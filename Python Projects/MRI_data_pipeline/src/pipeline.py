# src/pipeline.py
"""
Main pipeline orchestrator for dockerized skull stripping with watch mode.
"""
import argparse
import json
import time
import logging
from pathlib import Path
from typing import Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from utils import load_nifti, load_dicom_series, save_nifti, setup_logging
from preprocessing import preprocess_image
from registration import atlas_based_skull_strip
from quality_assessment import assess_quality, save_quality_report_json

logger = logging.getLogger(__name__)


def process_single_file(input_path: Path, config: dict, output_dir: Path):
    """Process a single MRI file or DICOM directory."""
    try:
        logger.info(f"Processing: {input_path.name}")

        # Load - handle both NIFTI files and DICOM directories
        if input_path.is_dir():
            img = load_dicom_series(input_path)
        else:
            img = load_nifti(input_path)
        
        # Preprocess
        normalize_method = config.get('normalize_method', 'zscore')
        preprocessed = preprocess_image(
            img,
            normalize_method=normalize_method,
            sigma=config.get('gaussian_sigma', 1.0)
        )

        # Determine mask target
        mask_target = config.get('mask_target', 'processed')
        
        if mask_target not in ['original', 'processed']:
            logger.warning(f"Invalid mask_target '{mask_target}', using 'processed'")
            mask_target = 'processed'
        
        # Skull strip with appropriate mask target
        if mask_target == 'original':
            logger.info("Mask will be applied to original (unprocessed) image")
            result = atlas_based_skull_strip(
                preprocessed,
                atlas_dir=Path(config['atlas_dir']),
                registration_type=config.get('registration_type', 'rigid'),
                normalize_method=normalize_method,
                mask_target='original',
                original_img_data=img
            )
        else:
            logger.info("Mask will be applied to preprocessed image")
            result = atlas_based_skull_strip(
                preprocessed,
                atlas_dir=Path(config['atlas_dir']),
                registration_type=config.get('registration_type', 'rigid'),
                normalize_method=normalize_method,
                mask_target='processed'
            )
        
        # Save result
        output_file = output_dir / f"{input_path.stem}_skull_stripped.nii.gz"
        save_nifti(result, output_file)
        logger.info(f"Saved result: {output_file.name}")

        # Quality assessment
        quality_results = assess_quality(result)

        # Save report as JSON
        report_file = output_dir / f"{input_path.stem}_quality_report.json"
        save_quality_report_json(quality_results, report_file, filename=input_path.name)

        logger.info(f"Saved quality report: {report_file.name}")

        # Create processing marker
        marker_file = output_dir / f".{input_path.name}.processed"
        marker_file.touch()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {input_path.name}: {e}", exc_info=True)

        # Create error marker
        error_file = output_dir / f".{input_path.name}.error"
        with open(error_file, 'w') as f:
            f.write(str(e))

        return False


def is_valid_nifti(filepath: Path) -> bool:
    """Check if file is a valid NIFTI file."""
    valid_extensions = ['.nii', '.nii.gz']
    return any(str(filepath).endswith(ext) for ext in valid_extensions)


def is_dicom_directory(dirpath: Path) -> bool:
    """Check if directory contains DICOM files."""
    if not dirpath.is_dir():
        return False

    # Check for common DICOM file extensions
    dicom_extensions = ['.dcm', '.dicom', '.DCM', '.DICOM']
    for ext in dicom_extensions:
        if any(dirpath.glob(f'*{ext}')):
            return True

    # Check for files without extension (common in DICOM)
    # Look for files that might be DICOM based on directory structure
    files = [f for f in dirpath.iterdir() if f.is_file()]
    if len(files) > 0:
        # If directory has multiple files and no obvious non-DICOM files,
        # it might be a DICOM series
        return True

    return False


def is_valid_input(input_path: Path) -> bool:
    """Check if input is a valid NIFTI file or DICOM directory."""
    return is_valid_nifti(input_path) or is_dicom_directory(input_path)


def is_already_processed(filepath: Path, output_dir: Path) -> bool:
    """Check if file has already been processed."""
    marker = output_dir / f".{filepath.name}.processed"
    error_marker = output_dir / f".{filepath.name}.error"
    return marker.exists() or error_marker.exists()


class MRIFileHandler(FileSystemEventHandler):
    """Handler for new MRI files and DICOM directories."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.processing: Set[Path] = set()

    def on_created(self, event):
        """Handle file and directory creation events."""
        input_path = Path(event.src_path)

        # Handle DICOM directories
        if event.is_directory:
            if not is_dicom_directory(input_path):
                return

            # Avoid processing same directory twice
            if input_path in self.processing:
                return

            # Wait for directory to be fully populated
            self._wait_for_directory_ready(input_path)

            # Check if already processed
            if is_already_processed(input_path, self.output_dir):
                logger.info(f"Skipping already processed directory: {input_path.name}")
                return

            # Process the directory
            self.processing.add(input_path)
            logger.info(f"New DICOM directory detected: {input_path.name}")

            success = process_single_file(input_path, self.config, self.output_dir)

            self.processing.discard(input_path)

            if success:
                logger.info(f"Successfully processed: {input_path.name}")
            else:
                logger.error(f"Failed to process: {input_path.name}")

            return

        # Handle NIFTI files
        if not is_valid_nifti(input_path):
            return

        # Avoid processing same file twice
        if input_path in self.processing:
            return

        # Wait for file to be fully written
        self._wait_for_file_ready(input_path)

        # Check if already processed
        if is_already_processed(input_path, self.output_dir):
            logger.info(f"Skipping already processed file: {input_path.name}")
            return

        # Process the file
        self.processing.add(input_path)
        logger.info(f"New file detected: {input_path.name}")

        success = process_single_file(input_path, self.config, self.output_dir)

        self.processing.discard(input_path)

        if success:
            logger.info(f"Successfully processed: {input_path.name}")
        else:
            logger.error(f"Failed to process: {input_path.name}")
    
    def _wait_for_file_ready(self, filepath: Path, timeout: int = 30):
        """Wait until file is fully written (size stops changing)."""
        if not filepath.exists():
            return

        start_time = time.time()
        prev_size = -1

        while time.time() - start_time < timeout:
            try:
                curr_size = filepath.stat().st_size
                if curr_size == prev_size and curr_size > 0:
                    # Size stable, file ready
                    time.sleep(1)  # Extra safety margin
                    return
                prev_size = curr_size
                time.sleep(0.5)
            except OSError:
                time.sleep(0.5)

        logger.warning(f"Timeout waiting for {filepath.name} to be ready")

    def _wait_for_directory_ready(self, dirpath: Path, timeout: int = 60):
        """Wait until directory is fully populated (file count stops changing)."""
        if not dirpath.exists():
            return

        start_time = time.time()
        prev_count = -1

        while time.time() - start_time < timeout:
            try:
                files = list(dirpath.iterdir())
                curr_count = len(files)
                if curr_count == prev_count and curr_count > 0:
                    # Count stable, directory ready
                    time.sleep(2)  # Extra safety margin for DICOM series
                    return
                prev_count = curr_count
                time.sleep(1)
            except OSError:
                time.sleep(1)

        logger.warning(f"Timeout waiting for directory {dirpath.name} to be ready")


def run_watch_mode(config_path: Path, input_dir: Path, output_dir: Path):
    """Run pipeline in watch mode."""
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    setup_logging(config.get('log_level', 'INFO'))
    
    logger.info("="*60)
    logger.info("SKULL STRIPPING PIPELINE - WATCH MODE")
    logger.info("="*60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Mask target: {config.get('mask_target', 'processed')}")
    logger.info("")
    logger.info("Watching for new files...")
    logger.info("Press Ctrl+C to stop")
    logger.info("="*60)
    
    # Process existing files and directories first
    logger.info("Processing existing files and DICOM directories...")

    # Collect NIFTI files
    existing_files = [f for f in input_dir.glob('*.nii*') if is_valid_nifti(f)]

    # Collect DICOM directories (only immediate subdirectories)
    existing_dirs = [d for d in input_dir.iterdir() if is_dicom_directory(d)]

    all_inputs = existing_files + existing_dirs

    for input_path in all_inputs:
        if not is_already_processed(input_path, output_dir):
            input_type = "directory" if input_path.is_dir() else "file"
            logger.info(f"Found existing {input_type}: {input_path.name}")
            process_single_file(input_path, config, output_dir)

    logger.info("Finished processing existing inputs")
    logger.info("")
    
    # Set up file system watcher
    event_handler = MRIFileHandler(config, output_dir)
    observer = Observer()
    observer.schedule(event_handler, str(input_dir), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping watch mode...")
        observer.stop()
    
    observer.join()
    logger.info("Shutdown complete")


def run_batch_mode(config_path: Path, input_dir: Path, output_dir: Path):
    """Run pipeline in batch mode (process once and exit)."""

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    setup_logging(config.get('log_level', 'INFO'))

    logger.info("="*60)
    logger.info("SKULL STRIPPING PIPELINE - BATCH MODE")
    logger.info("="*60)
    logger.info(f"Mask target: {config.get('mask_target', 'processed')}")

    # Process all files and DICOM directories
    input_files = [f for f in input_dir.glob('*.nii*') if is_valid_nifti(f)]
    input_dirs = [d for d in input_dir.iterdir() if is_dicom_directory(d)]

    all_inputs = input_files + input_dirs

    if not all_inputs:
        logger.warning("No NIFTI files or DICOM directories found in input directory")
        return

    logger.info(f"Found {len(input_files)} NIFTI file(s) and {len(input_dirs)} DICOM directory(ies) to process")

    success_count = 0
    for input_path in all_inputs:
        if is_already_processed(input_path, output_dir):
            logger.info(f"Skipping already processed: {input_path.name}")
            continue

        if process_single_file(input_path, config, output_dir):
            success_count += 1

    logger.info("")
    logger.info(f"Batch processing complete: {success_count}/{len(all_inputs)} succeeded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MRI Skull Stripping Pipeline"
    )
    parser.add_argument(
        '--config', 
        type=Path, 
        default=Path('./data/config/config.json'),
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input-dir', 
        type=Path, 
        default=Path('./data/input'),
        help='Input directory containing MRI files'
    )
    parser.add_argument(
        '--output-dir', 
        type=Path, 
        default=Path('./data/output'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Run in watch mode (continuously monitor for new files)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.watch:
        run_watch_mode(args.config, args.input_dir, args.output_dir)
    else:
        run_batch_mode(args.config, args.input_dir, args.output_dir)