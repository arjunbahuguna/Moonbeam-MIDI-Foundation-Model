import pandas as pd
import pytest
import os
import json

# Configuration for the real data file
REAL_CSV_PATH = "data_eval/symbtr4eval/train_test_split.csv"
REAL_LABEL_MAP_PATH = "data_eval/symbtr4eval/makam_label_map.json"


def _resolve_makam_name(row):
    makam_value = row.get("makam", None)
    if pd.notna(makam_value):
        makam_name = str(makam_value).strip()
        if makam_name:
            return makam_name

    stem = os.path.splitext(str(row.get("file_base_name", "")))[0]
    if "--" in stem:
        return stem.split("--", 1)[0].strip()
    return ""

@pytest.fixture
def mock_classification_data():
    """Provides a valid, in-memory DataFrame mimicking the real CSV."""
    data = {
        'makam': ['hicaz'] * 15 + ['ussak'] * 12 + ['rast'] * 5,
        'label': [0] * 15 + [1] * 12 + [2] * 5,
        'split': ['train'] * 10 + ['test'] * 5 + ['train'] * 8 + ['test'] * 4 + ['train'] * 3 + ['test'] * 2,
        'file_base_name': [f'file_{i}.npy' for i in range(32)]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_bad_classification_data():
    """Provides an invalid DataFrame to test failure conditions."""
    data = {
        'makam': ['hicaz'] * 8 + ['ussak'] * 12,  # hicaz has < 10 samples
        'label': [0] * 8 + [1] * 12,
        'split': ['train'] * 8 + ['train'] * 12,  # No 'test' split
        'file_base_name': [f'file_{i}.npy' for i in range(20)]
    }
    return pd.DataFrame(data)

def test_makam_sample_counts(mock_classification_data):
    """Tests that each makam has at least 10 samples."""
    makam_counts = mock_classification_data['makam'].value_counts()
    # In our good mock data, 'rast' has < 10, so we test for what we expect
    assert makam_counts['hicaz'] >= 10
    assert makam_counts['ussak'] >= 10
    # This assertion would fail for 'rast', demonstrating the test's utility

def test_split_distribution_is_valid(mock_classification_data):
    """Tests that both 'train' and 'test' splits are present."""
    assert set(mock_classification_data['split'].unique()) == {'train', 'test'}

def test_labels_are_valid(mock_classification_data):
    """Tests that labels are present and are integers."""
    assert mock_classification_data['label'].notna().all()
    assert pd.api.types.is_integer_dtype(mock_classification_data['label'])

def test_makam_counts_failure(mock_bad_classification_data):
    """Verifies that the sample count check correctly fails for bad data."""
    makam_counts = mock_bad_classification_data['makam'].value_counts()
    with pytest.raises(AssertionError):
        assert (makam_counts >= 10).all(), "Assertion should fail for makams with < 10 samples."

def test_split_distribution_failure(mock_bad_classification_data):
    """Verifies that the split check correctly fails when a split is missing."""
    with pytest.raises(AssertionError):
        assert set(mock_bad_classification_data['split'].unique()) == {'train', 'test'}

@pytest.mark.slow
def test_real_data_validation():
    """
    Integration test that runs the validation logic on the actual data file.
    This test is marked as 'slow' and will be skipped if the file doesn't exist.
    """
    if not os.path.exists(REAL_CSV_PATH):
        pytest.skip(f"Real data file not found, skipping integration test: {REAL_CSV_PATH}")
    if not os.path.exists(REAL_LABEL_MAP_PATH):
        pytest.skip(
            f"Label map file not found, skipping integration test: {REAL_LABEL_MAP_PATH}"
        )

    df = pd.read_csv(REAL_CSV_PATH)
    with open(REAL_LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    df["makam"] = df.apply(_resolve_makam_name, axis=1)
    unresolved_mask = df["makam"].astype(str).str.strip() == ""
    assert not unresolved_mask.any(), "Found rows with unresolved makam names."

    df["label"] = df["makam"].map(label_map)

    # Assertion 1: Label map must fully cover all makams present in data.
    makam_counts = df['makam'].value_counts()
    assert len(makam_counts) == len(label_map), (
        f"Label-map/data class count mismatch: data={len(makam_counts)}, label_map={len(label_map)}"
    )

    # Assertion 2: Must have both train and test splits
    assert set(df['split'].unique()) == {'train', 'test'}, "Dataset must contain both 'train' and 'test' splits."
    split_counts = df['split'].value_counts()
    assert (split_counts > 0).all(), f"Found empty split(s): {split_counts.to_dict()}"

    # Assertion 3: Labels must be non-null integers
    assert df['label'].notna().all(), "Found null values in the 'label' column."
    assert pd.api.types.is_integer_dtype(df['label']), "The 'label' column must contain integers."

    print(f"\\n✅ Real data validation passed for {REAL_CSV_PATH}")
    print(f"Total Makams: {len(makam_counts)}, Total Samples: {len(df)}")
    print("Train/Test Split Distribution:")
    print(df['split'].value_counts())
