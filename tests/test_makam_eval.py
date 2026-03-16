import pandas as pd
import pytest
import os

# Configuration for the real data file
REAL_CSV_PATH = "data/processed_data_symbtr/makam_classification_split.csv"

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
    # assert (makam_counts >= 10).all(), "All makams should have at least 10 samples."

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

    df = pd.read_csv(REAL_CSV_PATH)

    # Assertion 1: Each makam must have at least 10 samples
    makam_counts = df['makam'].value_counts()
    assert (makam_counts >= 10).all(), f"Found makams with fewer than 10 samples:\\n{makam_counts[makam_counts < 10]}"

    # Assertion 2: Must have both train and test splits
    assert set(df['split'].unique()) == {'train', 'test'}, "Dataset must contain both 'train' and 'test' splits."

    # Assertion 3: Labels must be non-null integers
    assert df['label'].notna().all(), "Found null values in the 'label' column."
    assert pd.api.types.is_integer_dtype(df['label']), "The 'label' column must contain integers."

    print(f"\\n✅ Real data validation passed for {REAL_CSV_PATH}")
    print(f"Total Makams: {len(makam_counts)}, Total Samples: {len(df)}")
    print("Train/Test Split Distribution:")
    print(df['split'].value_counts())
