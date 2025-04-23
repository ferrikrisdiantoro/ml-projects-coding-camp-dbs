import pandas as pd
import logging
from sqlalchemy import create_engine
import gspread
from google.oauth2.service_account import Credentials

# Konfigurasi logging
logging.basicConfig(
    filename="etl_log.log", level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def store_to_csv(df, file_name='products_clean.csv'):
    try:
        df.to_csv(file_name, index=False)
        print(f"Data berhasil disimpan ke {file_name}")
    except Exception as e:
        logging.error(f"Gagal menyimpan ke CSV {file_name}: {e}")

def store_to_postgre(df, db_url):
    engine = None
    try:
        engine = create_engine(db_url)
        df.to_sql('fashions_data', engine, if_exists='replace', index=False)
        print("Data berhasil disimpan ke PostgreSQL.")
    except Exception as e:
        logging.error(f"Gagal menyimpan ke PostgreSQL: {e}")
    finally:
        if engine:
            engine.dispose()

def store_to_google_sheets(df, sheet_name, creds_json):
    try:
        creds = Credentials.from_service_account_file(
            creds_json,
            scopes=["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
        )
        client = gspread.authorize(creds)
        try:
            sheet = client.open(sheet_name).sheet1
        except Exception:
            sheet = client.create(sheet_name).sheet1
        sheet.clear()
        sheet.update([df.columns.tolist()] + df.values.tolist())
        print(f"Data berhasil disimpan ke Google Sheets: {sheet_name}")
    except Exception as e:
        logging.error(f"Gagal menyimpan ke Google Sheets: {e}")
