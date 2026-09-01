"""Tests for ML model training and inference."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.data_loader import generate_sample_dataset
from ml.preprocessing import clean_dataframe, encode_protocol

def test_generate_sample_dataset():
    df, labels = generate_sample_dataset(n_samples=500)
    assert len(df) == 500
    assert len(labels) == 500
    assert labels.nunique() == 2

def test_clean_dataframe():
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({'a': [1, 2, float('inf'), None, 3], 'b': [4,5,6,7,8]})
    cleaned = clean_dataframe(df)
    assert not cleaned.isnull().any().any()
    assert not (cleaned == float('inf')).any().any()

def test_encode_protocol():
    import pandas as pd
    df = pd.DataFrame({'protocol': ['TCP', 'UDP', 'ICMP', 'UNKNOWN']})
    encoded = encode_protocol(df)
    assert 'protocol_num' in encoded.columns
    assert encoded.loc[0, 'protocol_num'] == 0   # TCP
    assert encoded.loc[1, 'protocol_num'] == 1   # UDP
