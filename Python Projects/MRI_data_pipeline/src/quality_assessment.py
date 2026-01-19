"""
Quality assessment metrics for skull stripping evaluation.
"""
import logging
import numpy as np
from scipy import ndimage
from scipy.ndimage import sobel
from typing import Dict, Optional
import warnings
import json
from datetime import datetime

from utils import ImageData

logger = logging.getLogger(__name__)


def calculate_mask_coverage(img_data: ImageData) -> float:
    """
    Calculate percentage of non-zero voxels in the image.
    
    Args:
        img_data: Skull-stripped image data
        
    Returns:
        Percentage of brain voxels (0-100)
    """
    total_voxels = np.prod(img_data.shape)
    brain_voxels = np.sum(img_data.data > 0)
    coverage = (brain_voxels / total_voxels) * 100
    
    logger.info(f"Mask coverage: {coverage:.2f}% ({brain_voxels}/{total_voxels} voxels)")
    return coverage


def calculate_brain_volume(img_data: ImageData) -> float:
    """
    Calculate brain volume in cm³ using voxel spacing.
    
    Args:
        img_data: Skull-stripped image data
        
    Returns:
        Brain volume in cm³
    """
    # Extract voxel spacing from affine matrix (mm)
    voxel_dims = np.abs(np.diag(img_data.affine[:3, :3]))
    voxel_volume_mm3 = np.prod(voxel_dims)
    voxel_volume_cm3 = voxel_volume_mm3 / 1000.0  # Convert mm³ to cm³
    
    brain_voxels = np.sum(img_data.data > 0)
    volume_cm3 = brain_voxels * voxel_volume_cm3
    
    logger.info(f"Brain volume: {volume_cm3:.2f} cm³")
    logger.info(f"Voxel dimensions: {voxel_dims} mm")
    
    return volume_cm3


def check_connected_components(img_data: ImageData) -> Dict[str, any]:
    """
    Analyze connected components in the brain mask.
    
    Args:
        img_data: Skull-stripped image data
        
    Returns:
        Dictionary with component analysis results
    """
    # Create binary mask
    binary_mask = img_data.data > 0
    
    # Label connected components
    labeled_array, num_features = ndimage.label(binary_mask)
    
    # Calculate size of each component
    component_sizes = ndimage.sum(binary_mask, labeled_array, 
                                   range(1, num_features + 1))
    
    if num_features > 0:
        largest_component_size = np.max(component_sizes)
        largest_component_fraction = largest_component_size / np.sum(binary_mask)
    else:
        largest_component_size = 0
        largest_component_fraction = 0
    
    results = {
        'num_components': int(num_features),
        'largest_component_size': int(largest_component_size),
        'largest_component_fraction': float(largest_component_fraction),
        'component_sizes': [int(size) for size in component_sizes] if len(component_sizes) > 0 else []
    }
    
    logger.info(f"Connected components: {num_features}")
    logger.info(f"Largest component: {largest_component_fraction*100:.1f}% of total")
    
    if num_features > 1:
        logger.warning(f"Multiple components detected ({num_features}). "
                      f"Brain should typically be one connected region.")
    
    return results


def calculate_edge_density(img_data: ImageData) -> float:
    """
    Calculate edge density at the brain boundary using Sobel filter.
    
    Args:
        img_data: Skull-stripped image data
        
    Returns:
        Average edge magnitude at boundary
    """
    # Create binary mask
    binary_mask = img_data.data > 0
    
    # Find boundary voxels (mask edge)
    eroded = ndimage.binary_erosion(binary_mask)
    boundary = binary_mask & ~eroded
    
    # Calculate gradient magnitude using Sobel operator
    sx = sobel(img_data.data, axis=0)
    sy = sobel(img_data.data, axis=1)
    sz = sobel(img_data.data, axis=2)
    gradient_magnitude = np.sqrt(sx**2 + sy**2 + sz**2)
    
    # Calculate average edge strength at boundary
    boundary_voxels = np.sum(boundary)
    if boundary_voxels > 0:
        edge_density = np.sum(gradient_magnitude[boundary]) / boundary_voxels
    else:
        edge_density = 0
    
    logger.info(f"Edge density at boundary: {edge_density:.4f}")
    
    return edge_density


def calculate_intensity_statistics(img_data: ImageData) -> Dict[str, float]:
    """
    Calculate intensity statistics for the brain region.
    
    Args:
        img_data: Skull-stripped image data
        
    Returns:
        Dictionary with intensity statistics
    """
    brain_voxels = img_data.data[img_data.data > 0]
    
    if len(brain_voxels) == 0:
        logger.warning("No brain voxels found!")
        return {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'median': 0,
            'q25': 0,
            'q75': 0
        }
    
    stats = {
        'mean': float(np.mean(brain_voxels)),
        'std': float(np.std(brain_voxels)),
        'min': float(np.min(brain_voxels)),
        'max': float(np.max(brain_voxels)),
        'median': float(np.median(brain_voxels)),
        'q25': float(np.percentile(brain_voxels, 25)),
        'q75': float(np.percentile(brain_voxels, 75))
    }
    
    logger.info(f"Intensity statistics:")
    logger.info(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    logger.info(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    logger.info(f"  Median: {stats['median']:.2f}")
    
    return stats

def calculate_dice_metrics(pred: ImageData, gt: ImageData) -> Dict[str, float]:
    """
    Calculate Dice coefficient and related metrics comparing prediction to ground truth.

    Args:
        pred: Predicted segmentation (skull-stripped image)
        gt: Ground truth segmentation (manual skull strip)

    Returns:
        Dictionary with Dice, Jaccard, sensitivity, precision
    """
    pred_bin = pred.data > 0
    gt_bin = gt.data > 0

    intersection = np.sum(pred_bin & gt_bin)
    pred_sum = np.sum(pred_bin)
    gt_sum = np.sum(gt_bin)
    union = np.sum(pred_bin | gt_bin)

    dice = 2.0 * intersection / (pred_sum + gt_sum + 1e-8)
    jaccard = intersection / (union + 1e-8)
    sensitivity = intersection / (gt_sum + 1e-8)  # Recall / True Positive Rate
    precision = intersection / (pred_sum + 1e-8)  # Positive Predictive Value

    metrics = {
        'dice': float(dice),
        'jaccard': float(jaccard),
        'sensitivity': float(sensitivity),
        'precision': float(precision),
        'intersection_voxels': int(intersection),
        'pred_voxels': int(pred_sum),
        'gt_voxels': int(gt_sum)
    }

    logger.info(f"Dice coefficient: {dice:.4f}")
    logger.info(f"Jaccard index: {jaccard:.4f}")
    logger.info(f"Sensitivity: {sensitivity:.4f}, Precision: {precision:.4f}")

    return metrics


def assess_quality(img_data: ImageData, 
                   ground_truth_mask: Optional[ImageData] = None) -> Dict[str, any]:
    """
    Comprehensive quality assessment of skull-stripped image.
    
    Args:
        img_data: Skull-stripped image to assess
        ground_truth_mask: Optional manual/ground truth mask
        
    Returns:
        Dictionary with all quality metrics and pass/fail flags
    """
    logger.info("Starting quality assessment...")
    
    results = {}
    
    # 1. Mask coverage
    coverage = calculate_mask_coverage(img_data)
    results['mask_coverage_percent'] = coverage
    results['coverage_ok'] = 5.0 < coverage < 40.0  # Typical brain is 10-20% of volume
    
    # 2. Brain volume
    volume = calculate_brain_volume(img_data)
    results['brain_volume_cm3'] = volume
    results['volume_ok'] = 800 < volume < 2000  # Typical adult brain: 1000-1500 cm³
    
    # 3. Connected components
    components = check_connected_components(img_data)
    results['connected_components'] = components
    results['components_ok'] = components['num_components'] == 1
    
    # 4. Edge density
    edge_density = calculate_edge_density(img_data)
    results['edge_density'] = edge_density
    # Lower is better - smooth boundary
    results['edge_density_ok'] = edge_density < 50.0
    
    # 5. Intensity statistics
    intensity_stats = calculate_intensity_statistics(img_data)
    results['intensity_stats'] = intensity_stats
    results['intensity_ok'] = intensity_stats['std'] > 0.01  # Has variation
    
    # 6. Dice coefficient against ground truth (if provided)
    if ground_truth_mask is not None:
        dice_metrics = calculate_dice_metrics(img_data, ground_truth_mask)
        results['dice_metrics'] = dice_metrics
        results['dice_ok'] = dice_metrics['dice'] > 0.85  # Good segmentation > 0.85

    # Overall pass/fail
    checks = [
        results.get('coverage_ok', False),
        results.get('volume_ok', False),
        results.get('components_ok', False),
        results.get('edge_density_ok', False),
        results.get('intensity_ok', False)

    ]
    if ground_truth_mask is not None:
        checks.append(results.get('dice_ok', False))
    
    results['passed_checks'] = int(sum(checks))
    results['total_checks'] = int(len(checks))
    results['overall_pass'] = results['passed_checks'] >= (len(checks) - 1)  # Allow 1 failure
    
    logger.info(f"\nQuality Assessment Summary:")
    logger.info(f"  Passed {results['passed_checks']}/{results['total_checks']} checks")
    logger.info(f"  Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")
    
    return results



def format_quality_report_json(results: Dict[str, any],
                                filename: Optional[str] = None,
                                timestamp: Optional[str] = None) -> Dict:
    """
    Format quality assessment results as a structured JSON report.

    Args:
        results: Dictionary from assess_quality()
        filename: Optional filename being assessed
        timestamp: Optional timestamp string

    Returns:
        Structured dictionary ready for JSON serialization
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    report = {
        "metadata": {
            "report_version": "1.0",
            "generated_at": timestamp or datetime.now().isoformat(),
            "filename": filename
        },
        "summary": {
            "overall_status": "PASS" if results['overall_pass'] else "FAIL",
            "checks_passed": int(results['passed_checks']),
            "total_checks": int(results['total_checks'])
        },
        "metrics": {
            "mask_coverage": {
                "value": round(results['mask_coverage_percent'], 2),
                "unit": "percent",
                "status": "PASS" if results['coverage_ok'] else "FAIL",
                "threshold": "5.0 < value < 40.0",
                "description": "Percentage of non-zero voxels"
            },
            "brain_volume": {
                "value": round(results['brain_volume_cm3'], 2),
                "unit": "cm3",
                "status": "PASS" if results['volume_ok'] else "FAIL",
                "threshold": "800 < value < 2000",
                "description": "Estimated brain volume"
            },
            "connected_components": {
                "count": int(results['connected_components']['num_components']),
                "largest_component_fraction": round(
                    float(results['connected_components']['largest_component_fraction']), 4
                ),
                "largest_component_size": int(results['connected_components']['largest_component_size']),
                "status": "PASS" if results['components_ok'] else "FAIL",
                "threshold": "count == 1",
                "description": "Number of disconnected brain regions"
            },
            "edge_density": {
                "value": round(results['edge_density'], 4),
                "unit": "arbitrary",
                "status": "PASS" if results['edge_density_ok'] else "FAIL",
                "threshold": "value < 50.0",
                "description": "Average edge magnitude at brain boundary"
            },
            "intensity_statistics": {
                "mean": round(results['intensity_stats']['mean'], 2),
                "std": round(results['intensity_stats']['std'], 2),
                "min": round(results['intensity_stats']['min'], 2),
                "max": round(results['intensity_stats']['max'], 2),
                "median": round(results['intensity_stats']['median'], 2),
                "q25": round(results['intensity_stats']['q25'], 2),
                "q75": round(results['intensity_stats']['q75'], 2),
                "status": "PASS" if results['intensity_ok'] else "FAIL",
                "threshold": "std > 0.01",
                "description": "Brain region intensity distribution"
            }
        }
    }

    if 'dice_metrics' in results:
        dice = results['dice_metrics']
        report['metrics']['ground_truth_comparison'] = {
            "dice": round(dice['dice'], 4),
            "jaccard": round(dice['jaccard'], 4),
            "sensitivity": round(dice['sensitivity'], 4),
            "precision": round(dice['precision'], 4),
            "intersection_voxels": dice['intersection_voxels'],
            "pred_voxels": dice['pred_voxels'],
            "gt_voxels": dice['gt_voxels'],
            "status": "PASS" if results.get('dice_ok', False) else "FAIL",
            "threshold": "dice > 0.85",
            "description": "Comparison against manual segmentation ground truth"
        }

    return report


def save_quality_report_json(results: Dict[str, any],
                              output_path: str,
                              filename: Optional[str] = None) -> None:
    """
    Save quality assessment report as a JSON file.

    Args:
        results: Dictionary from assess_quality()
        output_path: Path to save JSON file
        filename: Optional filename being assessed
    """
    report = format_quality_report_json(results, filename)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Quality report saved to: {output_path}")


def print_quality_report(results: Dict[str, any],
                        filename: Optional[str] = None,
                        output_format: str = 'text') -> None:
    """
    Print a formatted quality assessment report.

    Args:
        results: Dictionary from assess_quality()
        filename: Optional filename being assessed
        output_format: 'text' for human-readable, 'json' for JSON format
    """
    if output_format == 'json':
        report = format_quality_report_json(results, filename)
        print(json.dumps(report, indent=2))
        return

    # Original text format
    print("\n" + "="*60)
    print("QUALITY ASSESSMENT REPORT")
    if filename:
        print(f"File: {filename}")
    print("="*60)

    print(f"\n1. Mask Coverage: {results['mask_coverage_percent']:.2f}%")
    print(f"   Status: {'PASS' if results['coverage_ok'] else 'FAIL'}")

    print(f"\n2. Brain Volume: {results['brain_volume_cm3']:.2f} cm³")
    print(f"   Status: {'PASS' if results['volume_ok'] else 'FAIL'}")
    print(f"   Expected: 800-2000 cm³")

    comp = results['connected_components']
    print(f"\n3. Connected Components: {comp['num_components']}")
    print(f"   Largest component: {comp['largest_component_fraction']*100:.1f}%")
    print(f"   Status: {'PASS' if results['components_ok'] else 'FAIL'}")

    print(f"\n4. Edge Density: {results['edge_density']:.4f}")
    print(f"   Status: {'PASS' if results['edge_density_ok'] else 'FAIL'}")

    stats = results['intensity_stats']
    print(f"\n5. Intensity Statistics:")
    print(f"   Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    print(f"   Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    print(f"   Status: {'PASS' if results['intensity_ok'] else 'FAIL'}")

    if 'dice_metrics' in results:
        dice = results['dice_metrics']
        check_num = 6
        print(f"\n{check_num}. Ground Truth Comparison:")
        print(f"   Dice Coefficient: {dice['dice']:.4f}")
        print(f"   Jaccard Index: {dice['jaccard']:.4f}")
        print(f"   Sensitivity: {dice['sensitivity']:.4f}")
        print(f"   Precision: {dice['precision']:.4f}")
        print(f"   Intersection: {dice['intersection_voxels']} voxels")
        print(f"   Predicted: {dice['pred_voxels']} | Ground Truth: {dice['gt_voxels']}")
        print(f"   Status: {'PASS' if results['dice_ok'] else 'FAIL'}")
        print(f"   (Dice > 0.85 is good, > 0.9 is excellent)")

    print(f"\n" + "-"*60)
    print(f"Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")
    print(f"Passed {results['passed_checks']}/{results['total_checks']} checks")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    from utils import load_nifti, setup_logging
    from scrollview import Scroller

    setup_logging("INFO")

    # Load a skull-stripped image
    img_path = "./data/sample_data/processed/skull_stripped_final.nii"
    img = load_nifti(img_path)

    # Optional: Load ground truth mask for comparison
    # ground_truth_path = "/mask.nii"
    # ground_truth = load_nifti(ground_truth_path)

    # Run quality assessment
    results = assess_quality(img) # Add ground_truth_mask=ground_truth if available

    # Print report in text format (default)
    print("\n--- Text Format ---")
    print_quality_report(results, filename=img_path)

    # Print report in JSON format
    print("\n--- JSON Format ---")
    print_quality_report(results, filename=img_path, output_format='json')

    # Save as JSON file
    save_quality_report_json(results, "./quality_report_example.json", filename=img_path)
    print("\nJSON report saved to: ./quality_report_example.json")

    Scroller(img.data)