import time
import logging
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Konfigurasi logging
logging.basicConfig(
    filename="etl_log.log", level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    )
}

def fetching_content(url):
    session = requests.Session()
    try:
        response = session.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.text       # <-- gunakan .text agar unit test cek .lower() sukses
    except requests.exceptions.RequestException as e:
        logging.error(f"Terjadi kesalahan ketika request ke {url}: {e}")
        return None

def extract_fashion_data(fashion):
    if fashion is None:
        return None
    try:
        title = fashion.find('h3').text.strip()
        # Harga
        price = None
        pc = fashion.find('div', class_='price-container')
        if pc and pc.find('span', class_='price'):
            price = pc.find('span', class_='price').text.strip()

        # Rating
        rating = None
        rating_el = fashion.find('p', string=lambda t: t and "Rating:" in t)
        if rating_el:
            txt = rating_el.text.strip().replace("Rating:", "").strip()
            # ambil angka pertama
            import re
            m = re.search(r"(\d+(\.\d+)?)", txt)
            if m:
                rating = float(m.group(1))

        # Colors, Size, Gender
        colors = size = gender = None
        for p in fashion.find_all('p'):
            text = p.text.strip()
            # ambil angka atau teks setelah kata kunci, apapun polanya
            if "Colors" in text:
                # hapus kata 'Colors' dan titik dua, sisanya adalah nilai
                colors = text.replace("Colors", "").replace(":", "").strip()
            elif "Size" in text:
                size = text.replace("Size", "").replace(":", "").strip()
            elif "Gender" in text:
                gender = text.replace("Gender", "").replace(":", "").strip()

        return {
            "Title": title,
            "Price": price,
            "Rating": rating,
            "Colors": colors,
            "Size": size,
            "Gender": gender,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logging.error(f"Error saat ekstraksi data produk: {e}")
        return None

def scrape_fashion(base_url, start_page=1, delay=2):
    data = []
    page = start_page
    while True:
        url = f"{base_url}/page{page}" if page > 1 else base_url
        print(f"Scraping: {url}")
        html = fetching_content(url)
        if not html:
            break
        try:
            soup = BeautifulSoup(html, "html.parser")
            items = soup.find_all('div', class_='product-details')
            if not items:
                break
            for item in items:
                rec = extract_fashion_data(item)
                if rec:
                    data.append(rec)
            page += 1
            time.sleep(delay)
        except Exception as e:
            logging.error(f"Kesalahan saat scraping halaman {page}: {e}")
            break
    return data
