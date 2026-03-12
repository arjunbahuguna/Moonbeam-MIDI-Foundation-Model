import pandas as pd
import os


def main():
    # Configuration
    CSV_FILE = "processed_data_SymbTr/makam_classification_split.csv"

    print(f"Reading file: {CSV_FILE} ...")

    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: File '{CSV_FILE}' not found. Check the path.")
        return

    # Load data
    df = pd.read_csv(CSV_FILE)

    # Count samples per makam and label
    stats = df.groupby(['makam', 'label']).size().reset_index(name='count')

    # Sort by count descending
    stats = stats.sort_values(by='count', ascending=False)

    # Print results
    print("\n" + "=" * 50)
    print(f"{'Makam Name':<20} | {'Label ID':<10} | {'Total Samples':<10}")
    print("-" * 50)

    for _, row in stats.iterrows():
        print(f"{row['makam']:<20} | {row['label']:<10} | {row['count']:<10}")

    print("-" * 50)
    print(f"Total: {len(stats)} Makams, {len(df)} samples.")
    print("=" * 50)

    # Verify Train/Test split distribution
    print("\n📊 Train/Test Split Distribution:")
    split_counts = df['split'].value_counts()
    for split_name, count in split_counts.items():
        print(f" - {split_name}: {count} samples")


if __name__ == "__main__":
    main()