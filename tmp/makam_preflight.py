import os
import json
import pandas as pd
import numpy as np
import glob

def preflight():
    data_dir = "data_eval/symbtr4eval"
    csv_path = os.path.join(data_dir, "train_test_split.csv")
    label_map_path = os.path.join(data_dir, "makam_label_map.json")
    proc_dir = os.path.join(data_dir, "processed")
    config_path = "src/llama_recipes/configs/model_config_small_microtonal.json"

    # 1) Verify existence
    exists_csv = os.path.exists(csv_path)
    exists_label = os.path.exists(label_map_path)
    exists_proc = os.path.isdir(proc_dir)
    print(f"PRECHECK_CSV_EXISTS: {exists_csv}")
    print(f"PRECHECK_LABEL_MAP_EXISTS: {exists_label}")
    print(f"PRECHECK_PROC_DIR_EXISTS: {exists_proc}")

    if not (exists_csv and exists_label and exists_proc):
        print("PRECHECK_FAILURE: Essential files missing.")
        return

    # 2) Load CSV
    df = pd.read_csv(csv_path)
    print(f"PRECHECK_CSV_ROWS: {len(df)}")
    if "split" in df.columns:
        print(f"PRECHECK_SPLITS: {df['split'].value_counts().to_dict()}")
    
    required_cols = ["file_base_name", "makam"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    print(f"PRECHECK_MISSING_COLS: {missing_cols}")

    unique_makams = set(df["makam"].dropna().unique())
    print(f"PRECHECK_UNIQUE_MAKAMS_COUNT: {len(unique_makams)}")

    # 3) Load Label Map
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    print(f"PRECHECK_LABEL_MAP_COUNT: {len(label_map)}")
    
    map_keys = set(label_map.keys())
    diff_map = map_keys - unique_makams
    diff_csv = unique_makams - map_keys
    print(f"PRECHECK_LABEL_MISMATCH_MAP_MINUS_CSV: {list(diff_map)[:5]} (len={len(diff_map)})")
    print(f"PRECHECK_LABEL_MISMATCH_CSV_MINUS_MAP: {list(diff_csv)[:5]} (len={len(diff_csv)})")

    # 4) Check file coverage
    # Handling .npy extension in CSV
    def check_exists(x):
        if str(x).endswith(".npy"):
            path = os.path.join(proc_dir, x)
        else:
            path = os.path.join(proc_dir, f"{x}.npy")
        return os.path.exists(path)

    df['npy_exists'] = df['file_base_name'].apply(check_exists)
    missing_files = len(df[~df['npy_exists']])
    print(f"PRECHECK_MISSING_NPY_FILES: {missing_files}")
    
    duplicates = df['file_base_name'].duplicated().sum()
    print(f"PRECHECK_DUPLICATE_BASENAMES: {duplicates}")

    # 5) Label assignment consistency (if label col exists)
    if "label" in df.columns:
        df['expected_label'] = df['makam'].map(label_map)
        mismatches = (df['label'] != df['expected_label']).sum()
        print(f"PRECHECK_LABEL_COLS_MISMATCH: {mismatches}")

    # 6) Sample NPY files
    npy_files = glob.glob(os.path.join(proc_dir, "*.npy"))
    sample_size = min(200, len(npy_files))
    import random
    random.seed(42)
    sample_files = random.sample(npy_files, sample_size)
    
    vocab_limits = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            vocab_limits = {
                'onset': 1000000,
                'duration': cfg.get('num_duration_tokens', 4096),
                'octave': cfg.get('num_octave_tokens', 128),
                'pitch': cfg.get('num_pitch_tokens', 256),
                'instrument': cfg.get('num_instrument_tokens', 128),
                'velocity': cfg.get('num_velocity_tokens', 128)
            }
    
    shapes = set()
    dtypes = set()
    out_of_range = {col: 0 for col in ['onset', 'duration', 'octave', 'pitch', 'instrument', 'velocity']}
    
    for f in sample_files:
        data = np.load(f)
        if len(data.shape) > 1:
            shapes.add(data.shape[1])
            if data.shape[1] == 6:
                for i, col_name in enumerate(['onset', 'duration', 'octave', 'pitch', 'instrument', 'velocity']):
                    limit = vocab_limits.get(col_name, 999999)
                    col_data = data[:, i]
                    oor = np.sum((col_data < 0) | (col_data >= limit))
                    out_of_range[col_name] += int(oor)
        dtypes.add(str(data.dtype))

    print(f"PRECHECK_NPY_COLUMNS_COUNT: {list(shapes)}")
    print(f"PRECHECK_NPY_DTYPES: {list(dtypes)}")
    print(f"PRECHECK_NPY_OUT_OF_RANGE: {out_of_range}")

if __name__ == "__main__":
    preflight()
