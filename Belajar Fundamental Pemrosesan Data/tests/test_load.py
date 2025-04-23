import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from utils.load import store_to_csv, store_to_postgre, store_to_google_sheets

def test_store_to_csv(tmp_path):
    df = pd.DataFrame({'A': [1]})
    file = tmp_path / "out.csv"
    store_to_csv(df, str(file))
    assert file.exists()
    loaded = pd.read_csv(str(file))
    assert loaded['A'][0] == 1

def test_store_to_postgre(monkeypatch):
    df = pd.DataFrame({'A': [1]})
    fake_engine = MagicMock()
    # patch create_engine agar mengembalikan fake_engine
    monkeypatch.setattr('utils.load.create_engine', lambda url: fake_engine)
    # patch df.to_sql agar tidak error
    df.to_sql = MagicMock()
    store_to_postgre(df, 'dummy_url')
    # pastikan engine.dispose() dipanggil
    fake_engine.dispose.assert_called_once()

@patch('utils.load.Credentials.from_service_account_file')
@patch('utils.load.gspread')
def test_store_to_google_sheets_open_and_update(mock_gspread, mock_creds):
    df = pd.DataFrame({'A': [1]})
    mock_creds.return_value = object()
    fake_client = MagicMock()
    mock_gspread.authorize.return_value = fake_client

    # Simulasikan spreadsheet sudah ada
    fake_sheet = MagicMock()
    fake_client.open.return_value.sheet1 = fake_sheet

    store_to_google_sheets(df, 'sheet', 'json')
    fake_sheet.clear.assert_called_once()
    fake_sheet.update.assert_called_once()

@patch('utils.load.Credentials.from_service_account_file')
@patch('utils.load.gspread')
def test_store_to_google_sheets_create_and_update(mock_gspread, mock_creds):
    df = pd.DataFrame({'A': [1]})
    mock_creds.return_value = object()
    fake_client = MagicMock()
    mock_gspread.authorize.return_value = fake_client

    # Simulasikan spreadsheet tidak ditemukan → create dipanggil
    fake_client.open.side_effect = Exception('NotFound')
    fake_sheet = MagicMock()
    fake_client.create.return_value.sheet1 = fake_sheet

    store_to_google_sheets(df, 'sheet', 'json')
    fake_sheet.clear.assert_called_once()
    fake_sheet.update.assert_called_once()
