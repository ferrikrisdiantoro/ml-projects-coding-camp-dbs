# Laporan Proyek Machine Learning - Collaborative Filtering pada MovieLens
---
## Project Overview
---
Dalam era digital yang semakin berkembang, sistem rekomendasi memiliki peranan penting dalam membantu pengguna menavigasi informasi dan pilihan produk yang sangat besar. Salah satu aplikasi populer dari sistem rekomendasi adalah pada layanan streaming atau platform ulasan film. Dengan semakin banyaknya pilihan film dan meningkatnya kebutuhan personalisasi pengalaman pengguna, sistem rekomendasi yang efektif menjadi solusi penting untuk meningkatkan engagement dan kepuasan pengguna. Untuk itu, proyek ini bertujuan untuk membangun sistem rekomendasi berbasis collaborative filtering menggunakan dataset [MovieLens](https://grouplens.org/datasets/movielens/).

Dataset MovieLens merupakan salah satu benchmark yang paling sering digunakan dalam riset sistem rekomendasi. Teknik collaborative filtering berbasis neural network telah banyak diteliti sebagai metode untuk mempelajari preferensi pengguna. Misalnya, penelitian oleh [Koren, Bell, dan Volinsky (2009) dalam “Matrix Factorization Techniques for Recommender Systems”](https://ieeexplore.ieee.org/abstract/document/5197422) menunjukkan bagaimana pendekatan matrix factorization dapat menghasilkan prediksi rating yang akurat. Begitu pula, riset dalam [“Deep Neural Networks for YouTube Recommendations” oleh Covington et al. (2016)](https://dl.acm.org/doi/abs/10.1145/2959100.2959190) menggambarkan bagaimana teknik berbasis embedding dapat meningkatkan performa rekomendasi di lingkungan skala besar.

Dalam proyek ini, model yang dikembangkan menggunakan teknik collaborative filtering berbasis neural network untuk memprediksi film yang kemungkinan besar akan disukai oleh pengguna, berdasarkan rating yang diberikan oleh pengguna lain yang memiliki selera serupa.

Referensi:
1. Koren, Yehuda, Robert Bell, and Chris Volinsky. "*Matrix factorization techniques for recommender systems.*" Computer 42.8 (2009): 30-37.
2. Covington, Paul, Jay Adams, and Emre Sargin. "*Deep neural networks for youtube recommendations.*" Proceedings of the 10th ACM conference on recommender systems. 2016.

## Business Understanding
---
### Problem Statements

- Bagaimana memprediksi rating film yang belum pernah ditonton oleh pengguna?
- Bagaimana memberikan rekomendasi film yang relevan dan dipersonalisasi untuk pengguna berdasarkan histori rating-nya?

### Goals

- Mengembangkan model collaborative filtering untuk memprediksi rating film oleh pengguna.
- Menghasilkan top-N rekomendasi film yang disesuaikan dengan preferensi pengguna.

### Solution Statements

- Menggunakan pendekatan matrix factorization melalui embedding user dan item untuk mempelajari representasi laten.
- Mengimplementasikan dua pendekatan:
  1. **Dot product** antara user dan item embedding untuk prediksi rating (mirip baseline MF).
  2. **Neural network (NeuMF)** yang menggabungkan pendekatan Matrix Factorization (MF) dan Multi-Layer Perceptron (MLP) untuk meningkatkan akurasi prediksi.
- Menggunakan TensorFlow sebagai kerangka kerja utama dalam membangun dan melatih model.

## Data Understanding
---
Dataset yang digunakan adalah **MovieLens 100K**, yang tersedia secara publik melalui [GroupLens Research](https://grouplens.org/datasets/movielens/). Dataset ini sering digunakan sebagai benchmark dalam sistem rekomendasi.

Dataset terdiri dari empat file utama:

- `ratings.csv`: berisi 100.000 data rating yang diberikan oleh 943 pengguna terhadap 1.682 film. Kolom: `userId`, `movieId`, `rating`, `timestamp`.
- `movies.csv`: berisi 1.682 data film, dengan kolom `movieId`, `title`, dan `genres`.
- `tags.csv`: mencakup data tag yang diberikan oleh pengguna. Tidak digunakan dalam proyek ini.
- `links.csv`: berisi relasi antara `movieId` dengan ID dari IMDb dan TMDb. Tidak digunakan dalam proyek ini.

### Fitur-fitur dalam `ratings.csv`:
- `userId`: ID unik untuk pengguna.
- `movieId`: ID unik untuk film.
- `rating`: Nilai antara 0.5 hingga 5.0 (dalam kelipatan 0.5).
- `timestamp`: Waktu pemberian rating dalam Unix Time.

### Kondisi Data
- Tidak ditemukan missing value pada file `ratings.csv`.
- Distribusi rating cenderung bias ke nilai tinggi, dengan rating 4.0 dan 5.0 paling sering muncul.
- Beberapa pengguna memberikan banyak rating, sementara beberapa lainnya hanya memberikan sedikit.

### Visualisasi Distribusi Data Rating
Grafik menunjukkan jumlah frekuensi untuk setiap nilai rating mulai dari 0.5 hingga 5.0. Terlihat bahwa rating 4 memiliki jumlah paling tinggi, diikuti oleh rating 3 dan 5, yang menandakan bahwa sebagian besar pengguna memberikan ulasan positif terhadap aplikasi yang mereka nilai.

![Distribusi](./assets/DR.png)


### Visualisasi Histogram Data Rating
Histogram menampilkan penyebaran rating dalam bentuk batang dengan warna biru muda. Pola distribusinya serupa dengan barplot, di mana rating 4 mendominasi, menunjukkan bahwa data memiliki kecenderungan ke arah nilai tinggi, meskipun rating rendah juga masih cukup muncul.

![Histogram](./assets/HR.png)

### Visualisasi Boxplot Data Rating
Boxplot menggambarkan ringkasan statistik dari rating, seperti nilai minimum, maksimum, median, dan outlier. Median berada di antara rating 3 dan 4, yang menunjukkan kecenderungan rating tinggi, sementara titik-titik di sisi kiri (rating rendah) menunjukkan adanya ulasan negatif yang dianggap sebagai outlier.

![Boxplot](./assets/BR.png)

## Data Preparation 
---
Beberapa langkah data preparation yang dilakukan antara lain:

1. **Encoding**: Mengubah userId dan movieId menjadi indeks integer agar dapat digunakan pada layer embedding. Layer embedding memerlukan input dalam bentuk indeks numerik yang dapat dipetakan ke vektor representasi laten.
2. **Normalisasi**: Rating dinormalisasi ke dalam skala 0-1 karena model menggunakan fungsi aktivasi sigmoid pada output layer. Ini penting agar output model berada dalam rentang yang sesuai dengan target.
3. **Split Dataset**: Data dibagi menjadi training set dan validation set untuk mengevaluasi performa model secara objektif dan mencegah overfitting.

Langkah ini dilakukan untuk memastikan data dapat digunakan secara optimal dalam model berbasis deep learning.

## Modeling
---

Dalam proyek ini dikembangkan dua pendekatan model rekomendasi berbasis collaborative filtering. Dua pendekatan model digunakan untuk membandingkan performa model klasik berbasis matrix factorization dengan model deep learning berbasis neural network, guna mengeksplorasi performa dan fleksibilitas masing-masing dalam konteks rekomendasi film.

### 1. **Matrix Factorization dengan Embedding (Baseline)**

Model ini menggunakan pendekatan klasik collaborative filtering yang diimplementasikan menggunakan layer embedding dalam TensorFlow.

**Arsitektur:**
- Dua buah embedding layer: satu untuk user dan satu untuk movie.
- Operasi **dot product** antar embedding untuk mendapatkan skor kecocokan.
- Tambahan bias dan aktivasi sigmoid agar output berada dalam rentang 0–1.

     ```python
    class RecommenderNet(tf.keras.Model):
        def __init__(self, num_users, num_movies, embedding_size, **kwargs):
            super(RecommenderNet, self).__init__(**kwargs)
            self.num_users = num_users
            self.num_movies = num_movies
            self.embedding_size = embedding_size
        
        # Embedding untuk user
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        
        # Embedding untuk movie
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        
        # Dot product antara user dan movie embedding
        dot_user_movie = tf.reduce_sum(user_vector * movie_vector, axis=1, keepdims=True)
        
        # Menambahkan bias
        x = dot_user_movie + user_bias + movie_bias
        
        # Aktivasi sigmoid untuk output antara 0 dan 1
        return tf.nn.sigmoid(x)
     ```

**Kelebihan:**
- Sederhana dan efisien.
- Cepat dalam proses training.
- Cocok sebagai baseline atau model awal.

**Kekurangan:**
- Hubungan user–item yang dihasilkan hanya linear.
- Tidak cukup fleksibel untuk menangkap pola kompleks atau non-linear dalam preferensi pengguna.

### 2. **Neural Matrix Factorization (NeuMF)**

NeuMF merupakan pendekatan yang menggabungkan dua jalur: Generalized Matrix Factorization (GMF) dan Multi-Layer Perceptron (MLP). Pendekatan ini lebih fleksibel karena memungkinkan hubungan non-linear antar embedding.

**Arsitektur:**
- Embedding user dan item diproses melalui dua jalur:
  - **GMF**: menggunakan dot product seperti pada model klasik.
  - **MLP**: menggabungkan (concatenate) embedding dan melewati beberapa fully connected layer.
- Output kedua jalur digabung dan diproses oleh dense layer akhir.
- Aktivasi sigmoid pada output untuk menghasilkan skor prediksi.

     ```python
    def get_NeuMF_model(num_users, num_items, mf_dim=8, mlp_layers=[64,32,16,8], dropout=0.0):
        # Input layer
        user_input = Input(shape=(1,), name="user_input")
        item_input = Input(shape=(1,), name="item_input")

        # MF part embedding: gunakan embedding dimensi mf_dim
        mf_user_embedding = layers.Embedding(num_users, mf_dim, name="mf_user_embedding")(user_input)
        mf_item_embedding = layers.Embedding(num_items, mf_dim, name="mf_item_embedding")(item_input)
        mf_user_embedding = layers.Flatten()(mf_user_embedding)
        mf_item_embedding = layers.Flatten()(mf_item_embedding)
        mf_vector = layers.multiply([mf_user_embedding, mf_item_embedding])

        # MLP part embedding: gunakan ukuran embedding = mlp_layers[0]//2 agar jumlah dimensi tepat saat digabung
        mlp_embedding_dim = mlp_layers[0] // 2
        mlp_user_embedding = layers.Embedding(num_users, mlp_embedding_dim, name="mlp_user_embedding")(user_input)
        mlp_item_embedding = layers.Embedding(num_items, mlp_embedding_dim, name="mlp_item_embedding")(item_input)
        mlp_user_embedding = layers.Flatten()(mlp_user_embedding)
        mlp_item_embedding = layers.Flatten()(mlp_item_embedding)
        mlp_vector = layers.concatenate([mlp_user_embedding, mlp_item_embedding])
        
        # MLP layers
        for idx, units in enumerate(mlp_layers[1:]):
            mlp_vector = layers.Dense(units, activation='relu', name=f"mlp_dense_{idx}")(mlp_vector)
            if dropout > 0:
                mlp_vector = layers.Dropout(dropout)(mlp_vector)
        
        # Concatenate MF and MLP parts
        neumf_vector = layers.concatenate([mf_vector, mlp_vector])
        
        # Final prediction layer
        prediction = layers.Dense(1, activation="sigmoid", name="prediction")(neumf_vector)
        
        model = Model(inputs=[user_input, item_input], outputs=prediction)
        return model
     ```

**Kelebihan:**
- Dapat menangkap pola interaksi yang lebih kompleks.
- Fleksibel karena memanfaatkan kekuatan neural network dalam pembelajaran non-linear.

**Kekurangan:**
- Lebih kompleks dan lambat dibanding model klasik.
- Membutuhkan tuning hyperparameter yang lebih hati-hati.

### Top-N Recommendation

Dari hasil perbandingan, terlihat bahwa kedua model memiliki beberapa kesamaan dalam rekomendasi film, seperti *The Shawshank Redemption*, *Cinema Paradiso*, dan *Lawrence of Arabia*, yang menunjukkan adanya konsistensi terhadap film-film berkualitas tinggi. Namun, **NeuMF** cenderung merekomendasikan lebih banyak film klasik dan arthouse seperti *Paths of Glory* dan *Ran*, sedangkan **RecommenderNet** lebih condong pada film-film populer dan ikonik seperti *The Godfather* dan *Memento*. Hal ini menunjukkan bahwa NeuMF mampu menangkap preferensi yang lebih halus dan kompleks, sementara RecommenderNet memberikan hasil yang lebih general dan mainstream, sesuai karakteristik arsitektur masing-masing model.

Namun demikian, meskipun NeuMF memiliki diversifikasi genre yang lebih luas dalam hasil Top-N, hasil evaluasi kuantitatif seperti RMSE, MAE, dan R² menunjukkan bahwa RecommenderNet lebih akurat dalam memprediksi rating pengguna. Ini menandakan bahwa walaupun NeuMF menghasilkan rekomendasi yang terlihat "unik" atau "berbeda", ketepatan prediksi terhadap rating aktual tetap lebih baik pada RecommenderNet. Oleh karena itu, dalam konteks sistem rekomendasi berbasis rating prediction, RecommenderNet masih menjadi model yang lebih disarankan.

| Rank | **RecommenderNet**                                          | Predicted Score | Predicted Rating | **NeuMF**                                                | Predicted Score | Predicted Rating |
|------|---------------------------------------------------------------|------------------|-------------------|------------------------------------------------------------|------------------|-------------------|
| 1    | Shawshank Redemption, The (1994)                             | 0.9806           | 4.9127            | Shawshank Redemption, The (1994)                          | 0.9612           | 4.8253            |
| 2    | Godfather, The (1972)                                        | 0.9773           | 4.8977            | Dr. Strangelove or: How I Learned... (1964)              | 0.9536           | 4.7910            |
| 3    | Cinema Paradiso (Nuovo cinema Paradiso) (1989)              | 0.9706           | 4.8678            | Godfather, The (1972)                                     | 0.9533           | 4.7901            |
| 4    | 12 Angry Men (1957)                                          | 0.9631           | 4.8341            | Streetcar Named Desire, A (1951)                          | 0.9515           | 4.7819            |
| 5    | Lawrence of Arabia (1962)                                    | 0.9620           | 4.8292            | Cinema Paradiso (Nuovo cinema Paradiso) (1989)           | 0.9509           | 4.7792            |
| 6    | Godfather: Part II, The (1974)                               | 0.9617           | 4.8276            | Paths of Glory (1957)                                     | 0.9509           | 4.7791            |
| 7    | Patton (1970)                                                | 0.9611           | 4.8248            | Lawrence of Arabia (1962)                                 | 0.9500           | 4.7752            |
| 8    | Memento (2000)                                               | 0.9607           | 4.8232            | Ran (1985)                                                | 0.9497           | 4.7735            |
| 9    | Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)        | 0.9598           | 4.8192            | Inside Job (2010)                                         | 0.9493           | 4.7720            |
| 10   | Departed, The (2006)                                         | 0.9596           | 4.8182            | Three Billboards Outside Ebbing, Missouri (2017)         | 0.9491           | 4.7711            |


## Evaluation
---
Tujuan dari sistem rekomendasi ini adalah untuk memprediksi **rating** yang akan diberikan pengguna terhadap sebuah item (film) secara akurat. Oleh karena itu, model dievaluasi menggunakan metrik regresi yang mengukur seberapa dekat nilai prediksi dengan nilai aktual. Metrik Evaluasi yang Digunakan :

#### 1. Root Mean Squared Error (RMSE)
**RMSE** mengukur akar dari selisih kuadrat rata-rata antara nilai aktual ($y_i$) dan nilai prediksi ($\hat{y}_i$). Semakin kecil nilai **RMSE**, maka prediksi semakin mendekati nilai sebenarnya. **RMSE** bersifat sensitif terhadap outlier karena adanya pemangkatan kuadrat.. Dalam sistem rekomendasi berbasis rating seperti MovieLens, **RMSE** sering digunakan sebagai metrik utama karena penalti terhadap prediksi yang meleset jauh sangat penting.

**Rumus**:
  $$
  RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
  $$
  

#### 2. Mean Absolute Error (MAE)
**MAE** menghitung rata-rata dari selisih absolut antara nilai aktual dan prediksi. **MAE** tidak terlalu sensitif terhadap outlier, sehingga lebih “stabil” dibanding RMSE. **MAE** berguna untuk memahami rata-rata kesalahan prediksi tanpa pengaruh besar dari error ekstrem.

**Rumus**:
  $$
  MAE = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
  $$

#### 3. R-squared (R² Score)
**R²** mengukur seberapa besar proporsi variansi dari data aktual yang dapat dijelaskan oleh model. Nilainya berkisar antara:
  - **1.0** → prediksi sempurna  
  - **0.0** → model tidak lebih baik dari rata-rata  
  - **< 0** → model lebih buruk dari tebakan acak
Meskipun tidak umum digunakan sendiri untuk sistem rekomendasi, R² digunakan untuk melengkapi evaluasi secara statistik.

**Rumus**:
  $$
  R^2 = 1 - \frac{ \sum (y_i - \hat{y}_i)^2 }{ \sum (y_i - \bar{y})^2 }
  $$

### Hasil Evaluasi Metrik

| Metrik  | RecommenderNet | NeuMF    |
|---------|----------------|----------|
| RMSE    | 0.8529     | 0.8685   |
| MAE     | 0.6592     | 0.6660   |
| R²      | 0.3333     | 0.3088   |

Berdasarkan hasil evaluasi diatas, bisa disimpulkan bahwa:
- Model **RecommenderNet** menunjukkan performa **lebih unggul** dibandingkan NeuMF pada semua metrik.
- **RMSE dan MAE yang lebih rendah** mengindikasikan bahwa prediksi model lebih mendekati nilai aktual dan lebih stabil.
- **R² yang tinggi** menunjukkan bahwa model mampu menjelaskan sebagian besar variasi dalam data.

Berdasarkan hasil evaluasi menggunakan metrik RMSE, MAE, dan R², dapat disimpulkan bahwa model **RecommenderNet** mampu memberikan prediksi rating yang lebih akurat dibandingkan **NeuMF**, meskipun NeuMF menggunakan arsitektur yang lebih kompleks. Hal ini terlihat dari nilai RMSE dan MAE yang lebih rendah, serta nilai R² yang lebih tinggi pada RecommenderNet. Karena tujuan proyek adalah memberikan rekomendasi film yang tepat, penurunan RMSE dan MAE merupakan indikasi bahwa prediksi rating mendekati nilai aktual, sehingga rekomendasi yang dihasilkan juga relevan dan akurat. Dengan demikian, RecommenderNet lebih direkomendasikan untuk digunakan dalam sistem rekomendasi berbasis collaborative filtering pada studi ini. 

## Conclusion and Future Work
---
Dalam proyek ini, dua pendekatan collaborative filtering berhasil dibangun dan diuji menggunakan dataset MovieLens 100K. Hasil menunjukkan bahwa model RecommenderNet mampu memberikan hasil prediksi rating yang sedikit lebih baik dibandingkan NeuMF, meskipun NeuMF memiliki fleksibilitas yang lebih tinggi. Evaluasi menggunakan RMSE, MAE, dan R² Score memberikan gambaran bahwa model sudah cukup baik dalam merepresentasikan preferensi pengguna. 

Untuk pengembangan selanjutnya, model dapat diperluas dengan:
1. Menambahkan metrik evaluasi berbasis ranking (Hit@K, NDCG).
2. Menggabungkan pendekatan content-based untuk menangani cold start.
3. Melatih model pada data yang lebih besar dan kompleks seperti MovieLens 1M atau Netflix Prize dataset.