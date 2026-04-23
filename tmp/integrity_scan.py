import os
import pandas as pd
import numpy as np

csv_path = 'data_eval/symbtr4eval/train_test_split.csv'
processed_dir = 'data_eval/symbtr4eval/processed'

df = pd.read_csv(csv_path)
npy_files = df['file_base_name'].tolist()

total = len(npy_files)
exists_count = 0
valid_shape_count = 0
dtype_ok_count = 0
no_nan_count = 0
range_ok_count = 0
neg_onset_count = 0
onsets_min = float('inf')
onsets_max = float('-inf')

failures = []

# ranges: dur [0,1025], octave [0,12], pitch [0,1201], instrument [0,130], velocity [0,129]
# columns are assumed: 0:onset, 1:dur, 2:octave, 3:pitch, 4:instrument, 5:velocity
# Based on common Moonbeam/MIDI formats: (onset, dur, octave, pitch, instrument, velocity)
ranges = {
    1: (0, 1025), # dur
    2: (0, 12),   # octave
    3: (0, 1201), # pitch
    4: (0, 130),  # instrument
    5: (0, 129)   # velocity
}

for f_base in npy_files:
    f_path = os.path.join(processed_dir, f_base)
    if not os.path.exists(f_path):
        if len(failures) < 20:
            failures.append(f"{f_base}: file does not exist")
        continue
    exists_count += 1
    
    try:
        arr = np.load(f_path)
    except Exception as e:
        if len(failures) < 20:
            failures.append(f"{f_base}: error loading: {str(e)}")
        continue

    # array ndim==2 and second dim==6
    if arr.ndim != 2 or arr.shape[1] != 6:
        if len(failures) < 20:
            failures.append(f"{f_base}: invalid shape {arr.shape}")
        continue
    valid_shape_count += 1

    # dtype integer-like
    if not np.issubdtype(arr.dtype, np.integer) and not np.issubdtype(arr.dtype, np.floating):
         if len(failures) < 20:
            failures.append(f"{f_base}: unexpected dtype {arr.dtype}")
         continue
    
    # Check for NaN/Inf
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        if len(failures) < 20:
            failures.append(f"{f_base}: contains NaN or Inf")
        continue
    no_nan_count += 1
    
    # Integer-like check (if float, must be equal to its floor)
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(arr == np.floor(arr)):
            if len(failures) < 20:
                failures.append(f"{f_base}: contains non-integer values")
            continue
    dtype_ok_count += 1

    # Onset column 0
    onsets = arr[:, 0]
    onsets_min = min(onsets_min, np.min(onsets))
    onsets_max = max(onsets_max, np.max(onsets))
    neg_onset_count += np.sum(onsets < 0)

    # Range checks
    ok = True
    for col, (low, high) in ranges.items():
        vals = arr[:, col]
        if np.any(vals < low) or np.any(vals > high):
            if len(failures) < 20:
                failures.append(f"{f_base}: col {col} out of range [{low}, {high}]. Found min={np.min(vals)}, max={np.max(vals)}")
            ok = False
            break
    
    if ok:
        range_ok_count += 1

print(f"FULLSCAN_total_files: {total}")
print(f"FULLSCAN_exists: {exists_count}")
print(f"FULLSCAN_valid_shape: {valid_shape_count}")
print(f"FULLSCAN_no_nan_inf: {no_nan_count}")
print(f"FULLSCAN_dtype_integer_like: {dtype_ok_count}")
print(f"FULLSCAN_range_checks_passed: {range_ok_count}")
print(f"FULLSCAN_onset_min: {onsets_min}")
print(f"FULLSCAN_onset_max: {onsets_max}")
print(f"FULLSCAN_negative_onsets_count: {neg_onset_count}")

if failures:
    print("\nFULLSCAN_FAILURES:")
    for f in failures:
        print(f"FULLSCAN_FAIL: {f}")
