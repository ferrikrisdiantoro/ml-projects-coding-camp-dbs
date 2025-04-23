# 🐾 Proyek Klasifikasi Gambar Hewan dengan CNN

## 🚀 Setup Environment

### Menggunakan Anaconda
```bash
conda create --name animal-classifier python=3.9
conda activate animal-classifier
pip install -r requirements.txt
```

### Menggunakan Shell/Terminal
```bash
pipenv install
pipenv shell
pip install -r requirements.txt
```

## 🧠 Tentang Proyek
Proyek ini bertujuan untuk membangun model deep learning berbasis **Convolutional Neural Network (CNN)** untuk mengklasifikasikan gambar ke dalam 5 kategori hewan: **cat, dog, elephant, horse, dan lion**. Dataset yang digunakan terdiri dari **14.976 gambar** beresolusi beragam.

Model dikembangkan dengan TensorFlow/Keras dan di-deploy ke berbagai format: **SavedModel, TFLite, dan TensorFlow.js**, sehingga bisa digunakan di berbagai platform.

## 📦 Struktur Model

- ✅ Menggunakan `Sequential`, `Conv2D`, dan `Pooling Layer`
- ✅ Menggunakan `EarlyStopping`, `ModelCheckpoint`, dan `ReduceLROnPlateau`
- ✅ Akurasi tinggi (>98%) pada training dan testing
- ✅ Dataset dibagi menjadi **train**, **validation**, dan **test**

## 🖼️ Fitur Proyek

1. **Pelatihan CNN pada 5 Kelas Hewan**
   - Model dilatih untuk mengenali gambar kucing, anjing, gajah, kuda, dan singa.

2. **Evaluasi Akurasi dan Visualisasi Confusion Matrix**
   - Akurasi training: 99.22%  
   - Akurasi testing: 98.40%  
   - Confusion matrix menunjukkan performa yang sangat baik di semua kelas.

3. **Visualisasi Akurasi dan Loss**
   - Menampilkan grafik akurasi & loss selama training untuk mendeteksi overfitting/underfitting.

4. **Export Model ke SavedModel, TFLite, dan TFJS**
   - Model disimpan dalam tiga format berbeda untuk kebutuhan deployment multiplatform.

5. **Inference Langsung dari Gambar**
   - Menampilkan prediksi langsung dan probabilitas untuk setiap kelas dari gambar input.


## 📁 Struktur Folder
- `saved_model/` → format TensorFlow SavedModel
- `tflite/model.tflite` → format TensorFlow Lite
- `tfjs_model/` → format TensorFlow.js
- `Notebook.ipynb` → File notebook proyek
- `requirements.txt` → Depedensi library yang digunakan
