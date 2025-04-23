from utils.extract import scrape_fashion
from utils.transform import transform_data
from utils.load import store_to_csv, store_to_postgre, store_to_google_sheets
import pandas as pd

def main():
    BASE_URL = 'https://fashion-studio.dicoding.dev'
    EXCHANGE_RATE = 16000
    DB_URL = 'postgresql+psycopg2://developer:123Ferri!@localhost:5432/fashionsdb'
    SHEET_NAME = "Fashion_Data_Ferri"
    CREDS_JSON = "google-sheets-api.json"

    print(f"{10*'='} Mulai proses scraping{10*'='} ")
    data = scrape_fashion(BASE_URL)

    if data:
        print(f"{10*'.'}Berhasil ambil {len(data)} produk. Memproses transformasi data{10*'.'}")
        df = pd.DataFrame(data)
        df = transform_data(df, EXCHANGE_RATE)

        print(f"{10*'='}Menyimpan ke CSV dan database{10*'='}")
        store_to_csv(df, 'products.csv')
        store_to_postgre(df, DB_URL)
        
        print(f">>> DataFrame shape: {df.shape}") 
        print(df.head())
        
        print(f"{10*'='} Menyimpan data ke Google Sheets{10*'='}")
        store_to_google_sheets(df, SHEET_NAME, CREDS_JSON)
        print(f"{10*'.'} Data berhasil disimpan ke Google Sheets{10*'.'}")
    else:
        print("Tidak ada data yang berhasil diambil.")

if __name__ == '__main__':
    main()