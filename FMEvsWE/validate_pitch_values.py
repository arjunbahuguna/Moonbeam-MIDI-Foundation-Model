#!/usr/bin/env python3
"""
Pitch Value Validation Script for Microtonal CPT Experiment

This script scans the processed/ directory for .npy files and validates that
the pitch values (column 3, 0-indexed column 2) are in the expected range
of 0-1199 cents for microtonal data.

Usage:
    python validate_pitch_values.py --data_dir /path/to/processed
"""

import os
import argparse
import numpy as np
from pathlib import Path
from collections import Counter


def validate_pitch_values(data_dir: str) -> dict:
    """
    Validate pitch values in all .npy files in the processed directory.
    
    Args:
        data_dir: Path to the processed directory containing .npy files
        
    Returns:
        Dictionary with validation results
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all .npy files
    npy_files = list(data_path.glob("*.npy"))
    
    if not npy_files:
        print(f"WARNING: No .npy files found in {data_dir}")
        return {"error": "No .npy files found"}
    
    print(f"Found {len(npy_files)} .npy files in {data_dir}")
    print("=" * 60)
    
    all_pitch_values = []
    file_results = []
    files_with_microtonal = 0
    files_with_pitch_above_1100 = 0
    
    for npy_file in sorted(npy_files):
        try:
            # Load the .npy file
            data = np.load(npy_file)
            
            if data.ndim < 2 or data.shape[1] < 3:
                file_results.append({
                    "file": npy_file.name,
                    "error": f"Invalid shape: {data.shape}, expected at least 3 columns"
                })
                continue
            
            # Column 3 (0-indexed: column 2) contains pitch values in cents
            pitch_values = data[:, 2]
            
            min_pitch = pitch_values.min()
            max_pitch = pitch_values.max()
            unique_pitches = len(np.unique(pitch_values))
            
            # Check for microtonal pitches (not multiples of 100)
            western_pitches = {0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100}
            microtonal_mask = ~np.isin(pitch_values.astype(int), list(western_pitches))
            microtonal_count = microtonal_mask.sum()
            
            has_microtonal = microtonal_count > 0
            has_above_1100 = max_pitch > 1100
            
            if has_microtonal:
                files_with_microtonal += 1
            if has_above_1100:
                files_with_pitch_above_1100 += 1
            
            all_pitch_values.extend(pitch_values.tolist())
            
            file_results.append({
                "file": npy_file.name,
                "shape": data.shape,
                "min_pitch": int(min_pitch),
                "max_pitch": int(max_pitch),
                "unique_pitches": unique_pitches,
                "microtonal_count": int(microtonal_count),
                "has_microtonal": has_microtonal,
                "has_above_1100": has_above_1100,
            })
            
        except Exception as e:
            file_results.append({
                "file": npy_file.name,
                "error": str(e)
            })
    
    # Aggregate results
    all_pitch_values = np.array(all_pitch_values)
    overall_min = int(all_pitch_values.min())
    overall_max = int(all_pitch_values.max())
    overall_unique = len(np.unique(all_pitch_values))
    
    # Count pitch distribution
    pitch_counter = Counter(all_pitch_values.astype(int))
    
    # Western vs microtonal counts
    western_counts = sum(v for k, v in pitch_counter.items() if k in {0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100})
    microtonal_counts = sum(v for k, v in pitch_counter.items() if k not in {0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100})
    
    results = {
        "total_files": len(npy_files),
        "files_with_microtonal": files_with_microtonal,
        "files_with_pitch_above_1100": files_with_pitch_above_1100,
        "overall_min_pitch": overall_min,
        "overall_max_pitch": overall_max,
        "overall_unique_pitches": overall_unique,
        "total_pitch_tokens": len(all_pitch_values),
        "western_pitch_tokens": western_counts,
        "microtonal_pitch_tokens": microtonal_counts,
        "file_results": file_results,
    }
    
    return results


def print_report(results: dict):
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("PITCH VALUE VALIDATION REPORT")
    print("=" * 60)
    
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return
    
    print(f"\nTotal .npy files: {results['total_files']}")
    print(f"Files with microtonal pitches: {results['files_with_microtonal']}")
    print(f"Files with pitch > 1100 cents: {results['files_with_pitch_above_1100']}")
    
    print(f"\n--- Pitch Value Statistics ---")
    print(f"Overall min pitch: {results['overall_min_pitch']} cents")
    print(f"Overall max pitch: {results['overall_max_pitch']} cents")
    print(f"Overall unique pitch values: {results['overall_unique_pitches']}")
    print(f"Total pitch tokens: {results['total_pitch_tokens']}")
    print(f"Western pitch tokens (0,100,...,1100): {results['western_pitch_tokens']}")
    print(f"Microtonal pitch tokens: {results['microtonal_pitch_tokens']}")
    
    # Validation checks
    print(f"\n--- Validation Checks ---")
    
    # Check 1: Pitch values should be >= 0
    if results['overall_min_pitch'] >= 0:
        print("[PASS] All pitch values >= 0")
    else:
        print(f"[FAIL] Found pitch values < 0 (min: {results['overall_min_pitch']})")
    
    # Check 2: Pitch values should be <= 1199
    if results['overall_max_pitch'] <= 1199:
        print("[PASS] All pitch values <= 1199")
    else:
        print(f"[FAIL] Found pitch values > 1199 (max: {results['overall_max_pitch']})")
    
    # Check 3: Should have microtonal pitches
    if results['files_with_microtonal'] > 0:
        print(f"[PASS] Found microtonal pitches in {results['files_with_microtonal']} files")
    else:
        print("[WARN] No microtonal pitches found (all pitches are western)")
    
    # Check 4: Microtonal pitch ratio
    if results['total_pitch_tokens'] > 0:
        micro_ratio = results['microtonal_pitch_tokens'] / results['total_pitch_tokens'] * 100
        print(f"[INFO] Microtonal pitch ratio: {micro_ratio:.2f}%")
    
    # Print per-file summary (first 10 files)
    print(f"\n--- Per-File Summary (first 10 files) ---")
    for fr in results['file_results'][:10]:
        if 'error' in fr:
            print(f"  {fr['file']}: ERROR - {fr['error']}")
        else:
            print(f"  {fr['file']}: min={fr['min_pitch']}, max={fr['max_pitch']}, "
                  f"unique={fr['unique_pitches']}, micro={fr['microtonal_count']}")
    
    if len(results['file_results']) > 10:
        print(f"  ... and {len(results['file_results']) - 10} more files")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate pitch values in processed .npy files for microtonal CPT"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the processed directory containing .npy files"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional: Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    results = validate_pitch_values(args.data_dir)
    print_report(results)
    
    if args.output_json:
        import json
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            return obj
        
        serializable_results = convert_types(results)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
