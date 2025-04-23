# Dashboard Analisis Kualitas Udara 🌍

## 🚀 Setup Environment

### Menggunakan Anaconda
```bash
conda create --name air-quality python=3.9
conda activate air-quality
pip install -r requirements.txt
```

### Menggunakan Shell/Terminal
```bash
pipenv install
pipenv shell
pip install -r requirements.txt
```

## 📊 Menjalankan Aplikasi
```bash
cd dashboard
streamlit run dashboard.py
```

## 📌 Fitur Dashboard

1. **Tren Harian Rata-rata Polutan**  
   - Menampilkan tren harian PM2.5 dan PM10 dalam bentuk garis waktu.

2. **Distribusi Kualitas Udara per Lokasi**  
   - Boxplot untuk menganalisis distribusi kualitas udara di berbagai lokasi.

3. **Hubungan antara Parameter Cuaca dan Polusi**  
   - Heatmap untuk menunjukkan korelasi antara faktor cuaca dan polutan utama.

4. **Variasi Arah Angin dan Tingkat Polusi**  
   - Bar chart yang menunjukkan variasi polusi berdasarkan arah angin.

5. **Tren Kualitas Udara Berdasarkan Musim**  
   - Boxplot untuk menganalisis bagaimana polusi bervariasi di setiap musim.
