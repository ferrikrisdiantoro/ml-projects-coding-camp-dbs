# 📊 Predictive Analysis – Proyek Machine Learning Terapan  

Proyek ini bertujuan untuk membangun model **machine learning** yang mampu melakukan prediksi diabetes berdasarkan dataset tertentu. Model yang digunakan mencakup **Random Forest, XGBoost, dan LightGBM**, dengan evaluasi menggunakan metrik akurasi, F1-score, precision, dan recall.  

## 📂 Struktur Proyek  

- **Predictive_Analysis.ipynb** → Berisi eksplorasi data dan eksperimen model  
- **Laporan_Predictive_Analysis.md** → Laporan proyek  
- **diabetes_prediction_dataset.csv** → Dataset yang digunakan pada proyek  
- **requirements.txt** → Daftar dependensi proyek  
- **assets/images/** → Daftar Image yang digunakan untuk ditampilkan pada laporan proyek

## 🚀 Setup Environment  

### **Menggunakan Anaconda**  
```bash
conda create --name predictive-analysis python=3.9  
conda activate predictive-analysis  
pip install -r requirements.txt  
```  

### **Menggunakan Pipenv**  
```bash
pipenv install  
pipenv shell  
pip install -r requirements.txt  
```  

## 🎯 Tujuan Proyek  
1. Membandingkan performa beberapa model machine learning dalam memprediksi diabetes.  
2. Melakukan analisis hasil model untuk memilih yang paling optimal.  
3. Menyediakan dashboard interaktif untuk visualisasi hasil prediksi.  