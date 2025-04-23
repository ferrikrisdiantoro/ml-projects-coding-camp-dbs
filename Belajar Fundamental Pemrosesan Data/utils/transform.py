import pandas as pd
import logging
import re

# Konfigurasi logging
logging.basicConfig(
    filename="etl_log.log", level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def clean_price(price_str):
    if price_str is None:
        return None
    try:
        # unit test mengharapkan float('$100.00') → 100.0
        return float(price_str.replace('$', '').strip())
    except Exception as e:
        logging.error(f"Gagal membersihkan harga: {price_str} - {e}")
        return None

def clean_rating(rating_val):
    if rating_val is None:
        return None
    # coba langsung float
    try:
        return float(rating_val)
    except:
        txt = str(rating_val)
        m = re.search(r"(\d+(\.\d+)?)", txt)
        if m:
            try:
                return float(m.group(1))
            except Exception as e:
                logging.error(f"Gagal membersihkan rating: {rating_val} - {e}")
        return None

def clean_colors(color_str):
    try:
        return int(''.join(filter(str.isdigit, str(color_str))))
    except (ValueError, TypeError) as e:
        logging.error(f"Gagal membersihkan warna: {color_str} - {e}")
        return None
    
def clean_size(size_str):
    return str(size_str).strip() if size_str is not None else None

def clean_gender(gender_str):
    return str(gender_str).strip() if gender_str is not None else None

def transform_data(data, exchange_rate=16000):
    # terima list atau DataFrame
    if isinstance(data, list) and len(data) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()

    # filter invalid Title
    df = df[~df['Title'].isin([None, "", "Unknown Product"])].copy()

    # bersihkan kolom
    df['Price']  = df['Price'].apply(clean_price)
    df['Rating'] = df['Rating'].apply(clean_rating)
    df['Colors'] = df['Colors'].apply(clean_colors)
    df['Size']   = df['Size'].apply(clean_size)
    df['Gender'] = df['Gender'].apply(clean_gender)

    # konversi ke IDR dan drop baris tanpa Price
    df['Price'] = df['Price'] * exchange_rate
    df = df.dropna(subset=['Price'])
    df['Price'] = df['Price'].astype(float)

    # buang duplikat
    df = df.drop_duplicates()
    return df
