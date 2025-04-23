import pytest
from bs4 import BeautifulSoup
from utils.extract import fetching_content, extract_fashion_data, scrape_fashion

def test_fetching_content_valid():
    html = fetching_content('https://fashion-studio.dicoding.dev')
    assert html is not None
    assert '<html' in html.lower()

def test_fetching_content_invalid():
    # invalid URL → harus return None
    assert fetching_content('invalid-url') is None

def test_extract_fashion_data_valid():
    sample = '''
    <div class="product-details">
      <h3>Test</h3>
      <div class="price-container"><span class="price">$1.00</span></div>
      <p>Rating: 4.0⭐/5</p>
      <p>Colors: 3</p>
      <p>Size: M</p>
      <p>Gender: Unisex</p>
    </div>
    '''
    tag = BeautifulSoup(sample, 'html.parser').find('div', class_='product-details')
    data = extract_fashion_data(tag)
    assert isinstance(data, dict)
    assert data['Title'] == 'Test'
    assert data['Price'] == '$1.00'
    assert data['Rating'] == 4.0
    assert data['Colors'] == '3'
    assert data['Size'] == 'M'
    assert data['Gender'] == 'Unisex'
    assert 'Timestamp' in data

def test_extract_fashion_data_invalid_tag():
    # passing None atau tag tanpa struktur yang diharapkan → return None
    assert extract_fashion_data(None) is None
    empty_tag = BeautifulSoup('<div></div>', 'html.parser').find('div')
    assert extract_fashion_data(empty_tag) is None

def test_scrape_fashion_empty(monkeypatch):
    # patch fetching_content agar selalu mengembalikan halaman kosong
    monkeypatch.setattr('utils.extract.fetching_content', lambda url: '<html></html>')
    data = scrape_fashion('http://dummy', start_page=1, delay=0)
    assert data == []
