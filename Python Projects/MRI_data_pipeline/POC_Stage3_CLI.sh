#!/bin/bash
# POC_Stage3_CLI.sh - Demonstration of pipeline_CLI.py wrapper usage
# This script shows the simplified command-line interface matching the task specification

set -e  # Exit on error

echo "================================================================"
echo "STAGE 3: CLI WRAPPER DEMONSTRATION"
echo "================================================================"
echo ""
echo "This demonstrates the simplified CLI wrapper (pipeline_CLI.py)"
echo "which matches the task specification interface."
echo ""

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="./data/sample_data"
OUTPUT_DIR="./data/output_poc3_cli"
CONFIG_FILE="./config.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Setup:"
echo "  Input directory:  $INPUT_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Config file:      $CONFIG_FILE"
echo ""

# -----------------------------------------------------------------------------
# Example 1: Single file processing with defaults
# -----------------------------------------------------------------------------
echo "Example 1: Single file processing (basic usage)"
echo "----------------------------------------------------------------"
echo "This matches the task specification example:"
echo ""
echo "Command:"
echo "  python pipeline_CLI.py \\"
echo "    --input $INPUT_DIR/test_sample.nii \\"
echo "    --output $OUTPUT_DIR/example1_output.nii.gz"
echo ""

python pipeline_CLI.py \
  --input "$INPUT_DIR/test_sample.nii" \
  --output "$OUTPUT_DIR/example1_output.nii.gz"

echo ""
echo "✓ Example 1 complete"
echo ""
read -p "Press Enter to continue..."
echo ""

# -----------------------------------------------------------------------------
# Example 2: Single file with custom sigma parameter
# -----------------------------------------------------------------------------
echo "Example 2: Single file with custom smoothing parameter"
echo "----------------------------------------------------------------"
echo "Using sigma=1.5 for Gaussian smoothing"
echo ""
echo "Command:"
echo "  python pipeline_CLI.py \\"
echo "    --input $INPUT_DIR/test_sample.nii \\"
echo "    --output $OUTPUT_DIR/example2_output.nii.gz \\"
echo "    --sigma 1.5"
echo ""

python pipeline_CLI.py \
  --input "$INPUT_DIR/test_sample.nii" \
  --output "$OUTPUT_DIR/example2_output.nii.gz" \
  --sigma 1.5

echo ""
echo "✓ Example 2 complete"
echo ""
read -p "Press Enter to continue..."
echo ""

# -----------------------------------------------------------------------------
# Example 3: Single file with multiple custom parameters
# -----------------------------------------------------------------------------
echo "Example 3: Single file with custom normalization and registration"
echo "----------------------------------------------------------------"
echo "Using minmax normalization and affine registration"
echo ""
echo "Command:"
echo "  python pipeline_CLI.py \\"
echo "    --input $INPUT_DIR/test_sample.nii \\"
echo "    --output $OUTPUT_DIR/example3_output.nii.gz \\"
echo "    --normalize minmax \\"
echo "    --registration affine \\"
echo "    --sigma 2.0"
echo ""

python pipeline_CLI.py \
  --input "$INPUT_DIR/test_sample.nii" \
  --output "$OUTPUT_DIR/example3_output.nii.gz" \
  --normalize minmax \
  --registration affine \
  --sigma 2.0

echo ""
echo "✓ Example 3 complete"
echo ""
read -p "Press Enter to continue..."
echo ""

# -----------------------------------------------------------------------------
# Example 4: Apply mask to original image
# -----------------------------------------------------------------------------
echo "Example 4: Mask applied to original (unprocessed) image"
echo "----------------------------------------------------------------"
echo "Using --mask-target original to preserve original intensities"
echo ""
echo "Command:"
echo "  python pipeline_CLI.py \\"
echo "    --input $INPUT_DIR/test_sample.nii \\"
echo "    --output $OUTPUT_DIR/example4_output.nii.gz \\"
echo "    --mask-target original"
echo ""

python pipeline_CLI.py \
  --input "$INPUT_DIR/test_sample.nii" \
  --output "$OUTPUT_DIR/example4_output.nii.gz" \
  --mask-target original

echo ""
echo "✓ Example 4 complete"
echo ""
read -p "Press Enter to continue..."
echo ""

# -----------------------------------------------------------------------------
# Example 5: Batch processing directory
# -----------------------------------------------------------------------------
echo "Example 5: Batch processing directory"
echo "----------------------------------------------------------------"
echo "Processing all files in a directory"
echo ""

# Create input directory with multiple files
mkdir -p "$OUTPUT_DIR/example5/input"
cp "$INPUT_DIR/test_sample.nii" "$OUTPUT_DIR/example5/input/scan_001.nii"
cp "$INPUT_DIR/test_sample.nii" "$OUTPUT_DIR/example5/input/scan_002.nii"

echo "Command:"
echo "  python pipeline_CLI.py \\"
echo "    --input-dir $OUTPUT_DIR/example5/input \\"
echo "    --output-dir $OUTPUT_DIR/example5/output"
echo ""

python pipeline_CLI.py \
  --input-dir "$OUTPUT_DIR/example5/input" \
  --output-dir "$OUTPUT_DIR/example5/output"

echo ""
echo "✓ Example 5 complete"
echo ""
read -p "Press Enter to continue..."
echo ""

# -----------------------------------------------------------------------------
# Example 6: Batch processing with custom parameters
# -----------------------------------------------------------------------------
echo "Example 6: Batch processing with custom parameters"
echo "----------------------------------------------------------------"
echo "Processing directory with minmax normalization and sigma=1.5"
echo ""

mkdir -p "$OUTPUT_DIR/example6/input"
cp "$INPUT_DIR/test_sample.nii" "$OUTPUT_DIR/example6/input/scan_001.nii"
cp "$INPUT_DIR/test_sample.nii" "$OUTPUT_DIR/example6/input/scan_002.nii"

echo "Command:"
echo "  python pipeline_CLI.py \\"
echo "    --input-dir $OUTPUT_DIR/example6/input \\"
echo "    --output-dir $OUTPUT_DIR/example6/output \\"
echo "    --normalize minmax \\"
echo "    --sigma 1.5"
echo ""

python pipeline_CLI.py \
  --input-dir "$OUTPUT_DIR/example6/input" \
  --output-dir "$OUTPUT_DIR/example6/output" \
  --normalize minmax \
  --sigma 1.5

echo ""
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
echo "Command:"
echo "  python pipeline_CLI.py \\"
echo "    --input $INPUT_DIR/test_sample.nii \\"
echo "    --output $OUTPUT_DIR/example9_output.nii.gz \\"
echo "    --log-level DEBUG"
echo ""

python pipeline_CLI.py \
  --input "$INPUT_DIR/test_sample.nii" \
  --output "$OUTPUT_DIR/example9_output.nii.gz" \
  --log-level DEBUG

echo ""
echo "✓ Example 7 complete"
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "CLI WRAPPER DEMONSTRATION COMPLETE"
echo "================================================================"
echo ""
echo "Summary of examples:"
echo "  1. Basic single file processing"
echo "  2. Single file with custom sigma"
echo "  3. Single file with multiple custom parameters"
echo "  4. Mask applied to original image"
echo "  5. Batch processing directory"
echo "  6. Batch processing with custom parameters"
echo "  7. Debug-level logging"
echo ""
echo "All outputs saved to: $OUTPUT_DIR/"
echo "================================================================"