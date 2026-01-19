#!/bin/bash
# POC_Stage3.sh - Demonstration of pipeline.py command-line usage
# This script shows various ways to use the skull stripping pipeline

set -e  # Exit on error

echo "================================================================"
echo "STAGE 3: COMMAND-LINE PIPELINE DEMONSTRATION"
echo "================================================================"
echo ""

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="./data/sample_data"
OUTPUT_DIR="./data/output_poc3"
CONFIG_FILE="./data/config/config.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Setup:"
echo "  Input directory:  $INPUT_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Config file:      $CONFIG_FILE"
echo ""

# -----------------------------------------------------------------------------
# Example 1: Basic usage with defaults
# -----------------------------------------------------------------------------
echo "Example 1: Basic usage with default configuration"
echo "----------------------------------------------------------------"
echo "Command:"
echo "  python src/pipeline.py \\"
echo "    --config $CONFIG_FILE \\"
echo "    --input-dir $INPUT_DIR \\"
echo "    --output-dir $OUTPUT_DIR/example1"
echo ""

mkdir -p "$OUTPUT_DIR/example1/input"
cp "$INPUT_DIR/test_sample.nii" "$OUTPUT_DIR/example1/input/"

python src/pipeline.py \
  --config "$CONFIG_FILE" \
  --input-dir "$OUTPUT_DIR/example1/input" \
  --output-dir "$OUTPUT_DIR/example1"

echo "✓ Example 1 complete"
echo ""
read -p "Press Enter to continue..."
echo ""

# -----------------------------------------------------------------------------
# Example 2: Custom normalization and smoothing parameters
# -----------------------------------------------------------------------------
echo "Example 2: Custom preprocessing parameters"
echo "----------------------------------------------------------------"
echo "Using minmax normalization and sigma=2.0"
echo ""

# Create custom config
cat > "$OUTPUT_DIR/config_custom.json" << EOF
{
  "normalize_method": "minmax",
  "gaussian_sigma": 2.0,
  "registration_type": "rigid",
  "mask_target": "processed",
  "atlas_dir": "./MNI_atlas",
  "log_level": "INFO"
}
EOF

mkdir -p "$OUTPUT_DIR/example2/input"
cp "$INPUT_DIR/test_sample.nii" "$OUTPUT_DIR/example2/input/"

echo "Command:"
echo "  python src/pipeline.py \\"
echo "    --config $OUTPUT_DIR/config_custom.json \\"
echo "    --input-dir $OUTPUT_DIR/example2/input \\"
echo "    --output-dir $OUTPUT_DIR/example2"
echo ""

python src/pipeline.py \
  --config "$OUTPUT_DIR/config_custom.json" \
  --input-dir "$OUTPUT_DIR/example2/input" \
  --output-dir "$OUTPUT_DIR/example2"

echo "✓ Example 2 complete"
echo ""
read -p "Press Enter to continue..."
echo ""

# -----------------------------------------------------------------------------
# Example 3: Apply mask to original (unprocessed) image
# -----------------------------------------------------------------------------
echo "Example 3: Mask applied to original image"
echo "----------------------------------------------------------------"
echo "Processing with mask_target='original' to preserve original intensities"
echo ""

cat > "$OUTPUT_DIR/config_original_mask.json" << EOF
{
  "normalize_method": "zscore",
  "gaussian_sigma": 1.0,
  "registration_type": "rigid",
  "mask_target": "original",
  "atlas_dir": "./MNI_atlas",
  "log_level": "INFO"
}
EOF

mkdir -p "$OUTPUT_DIR/example3/input"
cp "$INPUT_DIR/test_sample.nii" "$OUTPUT_DIR/example3/input/"

echo "Command:"
echo "  python src/pipeline.py \\"
echo "    --config $OUTPUT_DIR/config_original_mask.json \\"
echo "    --input-dir $OUTPUT_DIR/example3/input \\"
echo "    --output-dir $OUTPUT_DIR/example3"
echo ""

python src/pipeline.py \
  --config "$OUTPUT_DIR/config_original_mask.json" \
  --input-dir "$OUTPUT_DIR/example3/input" \
  --output-dir "$OUTPUT_DIR/example3"

echo "✓ Example 3 complete"
echo ""
read -p "Press Enter to continue..."
echo ""

# -----------------------------------------------------------------------------
# Example 4: Watch mode simulation (process existing + monitor for new)
# -----------------------------------------------------------------------------
echo "Example 4: Watch mode (run for 30 seconds)"
echo "----------------------------------------------------------------"
echo "The pipeline will process existing files and watch for new ones"
echo ""

mkdir -p "$OUTPUT_DIR/example4/input"
cp "$INPUT_DIR/test_sample.nii" "$OUTPUT_DIR/example4/input/"

echo "Command:"
echo "  timeout 30 python src/pipeline.py \\"
echo "    --config $CONFIG_FILE \\"
echo "    --input-dir $OUTPUT_DIR/example4/input \\"
echo "    --output-dir $OUTPUT_DIR/example4 \\"
echo "    --watch"
echo ""
echo "Starting watch mode (will timeout after 10 seconds)..."

# Run in watch mode with timeout
timeout 30 python src/pipeline.py \
  --config "$CONFIG_FILE" \
  --input-dir "$OUTPUT_DIR/example4/input" \
  --output-dir "$OUTPUT_DIR/example4" \
  --watch || true

echo ""
echo "✓ Example 4 complete (watch mode timed out as expected)"
echo ""
read -p "Press Enter to continue..."
echo ""

# -----------------------------------------------------------------------------
# Example 5: Batch processing multiple files
# -----------------------------------------------------------------------------
echo "Example 5: Batch processing multiple files"
echo "----------------------------------------------------------------"
echo "Processing multiple files at once"
echo ""

mkdir -p "$OUTPUT_DIR/example5/input"
cp "$INPUT_DIR/test_sample.nii" "$OUTPUT_DIR/example5/input/scan_001.nii"
cp "$INPUT_DIR/test_sample.nii" "$OUTPUT_DIR/example5/input/scan_002.nii"

echo "Command:"
echo "  python src/pipeline.py \\"
echo "    --config $CONFIG_FILE \\"
echo "    --input-dir $OUTPUT_DIR/example5/input \\"
echo "    --output-dir $OUTPUT_DIR/example5"
echo ""

python src/pipeline.py \
  --config "$CONFIG_FILE" \
  --input-dir "$OUTPUT_DIR/example5/input" \
  --output-dir "$OUTPUT_DIR/example5"

echo "✓ Example 5 complete"
echo ""
read -p "Press Enter to continue..."
echo ""

# -----------------------------------------------------------------------------
# Example 6: Affine registration (more flexible than rigid)
# -----------------------------------------------------------------------------
echo "Example 6: Using affine registration"
echo "----------------------------------------------------------------"
echo "Affine registration allows scaling and shearing in addition to rotation/translation"
echo ""

cat > "$OUTPUT_DIR/config_affine.json" << EOF
{
  "normalize_method": "zscore",
  "gaussian_sigma": 1.0,
  "registration_type": "affine",
  "mask_target": "processed",
  "atlas_dir": "./MNI_atlas",
  "log_level": "INFO"
}
EOF

mkdir -p "$OUTPUT_DIR/example6/input"
cp "$INPUT_DIR/test_sample.nii" "$OUTPUT_DIR/example6/input/"

echo "Command:"
echo "  python src/pipeline.py \\"
echo "    --config $OUTPUT_DIR/config_affine.json \\"
echo "    --input-dir $OUTPUT_DIR/example6/input \\"
echo "    --output-dir $OUTPUT_DIR/example6"
echo ""

python src/pipeline.py \
  --config "$OUTPUT_DIR/config_affine.json" \
  --input-dir "$OUTPUT_DIR/example6/input" \
  --output-dir "$OUTPUT_DIR/example6"

echo "✓ Example 6 complete"
echo ""
read -p "Press Enter to continue..."
echo ""

# -----------------------------------------------------------------------------
# Example 7: Debug logging
# -----------------------------------------------------------------------------
echo "Example 7: Debug-level logging"
echo "----------------------------------------------------------------"
echo "Using DEBUG log level for detailed output"
echo ""

cat > "$OUTPUT_DIR/config_debug.json" << EOF
{
  "normalize_method": "zscore",
  "gaussian_sigma": 1.0,
  "registration_type": "rigid",
  "mask_target": "processed",
  "atlas_dir": "./MNI_atlas",
  "log_level": "DEBUG"
}
EOF

mkdir -p "$OUTPUT_DIR/example7/input"
cp "$INPUT_DIR/test_sample.nii" "$OUTPUT_DIR/example7/input/"

echo "Command:"
echo "  python src/pipeline.py \\"
echo "    --config $OUTPUT_DIR/config_debug.json \\"
echo "    --input-dir $OUTPUT_DIR/example7/input \\"
echo "    --output-dir $OUTPUT_DIR/example7"
echo ""

python src/pipeline.py \
  --config "$OUTPUT_DIR/config_debug.json" \
  --input-dir "$OUTPUT_DIR/example7/input" \
  --output-dir "$OUTPUT_DIR/example7"

echo "✓ Example 7 complete"
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "STAGE 3 DEMONSTRATION COMPLETE"
echo "================================================================"
echo ""
echo "Summary of examples:"
echo "  1. Basic usage with defaults"
echo "  2. Custom preprocessing parameters (minmax, sigma=2.0)"
echo "  3. Mask applied to original (unprocessed) image"
echo "  4. Watch mode for continuous processing"
echo "  5. Batch processing multiple files"
echo "  6. Affine registration"
echo "  7. Debug-level logging"
echo ""
echo "All outputs saved to: $OUTPUT_DIR/"
echo ""
echo "Quality reports (JSON) are available in each example's output directory"
echo ""
echo "Configuration parameters available:"
echo "  normalize_method  'zscore' or 'minmax'"
echo "  gaussian_sigma    Smoothing parameter (0.5 - 2.0)"
echo "  registration_type 'rigid' or 'affine'"
echo "  mask_target       'processed' or 'original'"
echo "  atlas_dir         Path to MNI atlas directory"
echo "  log_level         'DEBUG', 'INFO', 'WARNING', or 'ERROR'"
echo ""
echo "================================================================"