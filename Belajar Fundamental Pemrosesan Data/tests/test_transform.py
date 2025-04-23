import pytest
import pandas as pd
from utils.transform import clean_price, clean_rating, transform_data

def test_clean_price_usd_to_float():
    assert clean_price('$100.00') == 100.0

def test_clean_price_invalid():
    assert clean_price('invalid') is None
    assert clean_price(None) is None

def test_clean_rating_simple():
    assert clean_rating('4.5') == 4.5

def test_clean_rating_fallback():
    assert clean_rating('Rating: 3⭐/5') == 3.0

def test_clean_rating_invalid():
    assert clean_rating('abc') is None

def test_transform_data_empty():
    df = transform_data([], exchange_rate=16000)
    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_transform_data_conversion():
    data = [{
        'Title': 'X',
        'Price': '$1.00',
        'Rating': '4.0',
        'Colors': '3',
        'Size': 'M',
        'Gender': 'Unisex',
        'Timestamp': '2025-01-01 00:00:00'
    }]
    df = transform_data(data, exchange_rate=16000)
    # $1.00 → 1.0 → *16000 = 16000
    assert df.iloc[0]['Price'] == 16000
    assert df.iloc[0]['Rating'] == 4.0
    assert df.iloc[0]['Colors'] == 3
    assert df.iloc[0]['Size'] == 'M'
    assert df.iloc[0]['Gender'] == 'Unisex'
