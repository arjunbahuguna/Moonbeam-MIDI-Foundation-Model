import pandas as pd
import json
import os


def main():
    # Configuration
    CSV_FILE = "data/processed_data_symbtr/train_test_split.csv"
    MIN_TRAIN_PIECES = 10
    OUTPUT_CSV = "data/processed_data_symbtr/makam_classification_split.csv"
    OUTPUT_LABEL_MAP = "data/processed_data_symbtr/makam_label_map.json"

    print("Processing data...")

    # 1. Verify input file existence
    if not os.path.exists(CSV_FILE):
        print(f"Error: Input file '{CSV_FILE}' not found.")
        return

    # 2. Load data
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} records.")

    # 3. Calculate Makam frequency in training set
    train_df = df[df['split'] == 'train']
    makam_counts = train_df['makam'].value_counts()

    # Filter Makams with count >= threshold
    valid_makams = makam_counts[makam_counts >= MIN_TRAIN_PIECES].index.tolist()
    valid_makams.sort()

    print(f"Found {len(valid_makams)} valid Makams (count >= {MIN_TRAIN_PIECES}).")

    # 4. Filter dataset
    filtered_df = df[df['makam'].isin(valid_makams)].copy()

    # 5. Create label mapping (name -> index)
    label_map = {makam: i for i, makam in enumerate(valid_makams)}
    filtered_df['label'] = filtered_df['makam'].map(label_map)

    # 6. Ensure output directories exist
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_LABEL_MAP), exist_ok=True)

    # 7. Save results
    filtered_df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_LABEL_MAP, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)

    # Final statistics
    print("\n" + "=" * 30)
    print("Processing complete.")
    print(f"Total samples: {len(filtered_df)}")
    print(f"Train: {len(filtered_df[filtered_df['split'] == 'train'])}")
    print(f"Test: {len(filtered_df[filtered_df['split'] == 'test'])}")
    print(f"Classes (N): {len(valid_makams)}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Output Map: {OUTPUT_LABEL_MAP}")
    print("=" * 30)


if __name__ == "__main__":
    main()