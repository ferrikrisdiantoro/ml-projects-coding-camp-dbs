#!/usr/bin/env python
# coding: utf-8

# # **1. Import Library**

# In[1]:


# Aljabar linear dan Manipulasi Data
import pandas as pd  # Pandas untuk manipulasi dan analisis data
import numpy as np  # NumPy untuk komputasi numerik

# Label Sentimen Lexicon
import csv
import requests
from io import StringIO
 
# Teks Preprocessing
import datetime as dt  # Manipulasi data waktu dan tanggal
import re  # Modul untuk bekerja dengan ekspresi reguler
import string  # Berisi konstanta string, seperti tanda baca
from nltk.tokenize import word_tokenize  # Tokenisasi teks
from nltk.corpus import stopwords  # Daftar kata-kata berhenti dalam teks
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # Stemming (penghilangan imbuhan kata) dalam bahasa Indonesia
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory  # Menghapus kata-kata berhenti dalam bahasa Indonesia

# EDA dan Visualisasi
from collections import Counter
from wordcloud import WordCloud  # Membuat visualisasi berbentuk awan kata (word cloud) dari teks
import matplotlib.pyplot as plt  # Matplotlib untuk visualisasi data
import seaborn as sns  # Seaborn untuk visualisasi data statistik, mengatur gaya visualisasi

# Normalisasi dan Splitting
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#Ekstraksi Fitur
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

#avaluasi
from sklearn.metrics import classification_report, confusion_matrix

# NLP
import nltk  # Import pustaka NLTK (Natural Language Toolkit).
nltk.download('punkt')  # Mengunduh dataset yang diperlukan untuk tokenisasi teks.
nltk.download('stopwords')  # Mengunduh dataset yang berisi daftar kata-kata berhenti (stopwords) dalam berbagai bahasa.


# # **2. Memuat Dataset**

# In[2]:


path = '/mnt/d/Artificial Intellegence/Machine Learning/CodingCamp/Proyek Analisis Sentimen/Telkom_Reviews.csv'
df = pd.read_csv(path)


# In[3]:


df.head()


# In[4]:


df.info()


# # 3. Data Wrangling

# In[5]:


clean_df = df.dropna()


# In[6]:


# Menghapus baris duplikat dari DataFrame clean_df
clean_df = clean_df.drop_duplicates()
 
# Menghitung jumlah baris dan kolom dalam DataFrame clean_df setelah menghapus duplikat
jumlah_ulasan_setelah_hapus_duplikat, jumlah_kolom_setelah_hapus_duplikat = clean_df.shape


# In[7]:


clean_df.info()


# # 4. Text Preprocessing

# In[8]:


def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # menghapus mention
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # menghapus hashtag
    text = re.sub(r'RT[\s]', '', text) # menghapus RT
    text = re.sub(r"http\S+", '', text) # menghapus link
    text = re.sub(r'[0-9]+', '', text) # menghapus angka
    text = re.sub(r'[^\w\s]', '', text) # menghapus karakter selain huruf dan angka
 
    text = text.replace('\n', ' ') # mengganti baris baru dengan spasi
    text = text.translate(str.maketrans('', '', string.punctuation)) # menghapus semua tanda baca
    text = text.strip(' ') # menghapus karakter spasi dari kiri dan kanan teks
    return text
 
def casefoldingText(text): # Mengubah semua karakter dalam teks menjadi huruf kecil
    text = text.lower()
    return text
 
def tokenizingText(text): # Memecah atau membagi string, teks menjadi daftar token
    text = word_tokenize(text)
    return text
 
def filteringText(text): # Menghapus stopwords dalam teks
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords1 = set(stopwords.words('english'))
    listStopwords.update(listStopwords1)
    listStopwords.update(['iya','yaa','gak','nya','na','sih','ku',"di","ga","ya","gaa","loh","kah","woi","woii","woy"])
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text
 
def stemmingText(text): # Mengurangi kata ke bentuk dasarnya yang menghilangkan imbuhan awalan dan akhiran atau ke akar kata
    # Membuat objek stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
 
    # Memecah teks menjadi daftar kata
    words = text.split()
 
    # Menerapkan stemming pada setiap kata dalam daftar
    stemmed_words = [stemmer.stem(word) for word in words]
 
    # Menggabungkan kata-kata yang telah distem
    stemmed_text = ' '.join(stemmed_words)
 
    return stemmed_text
 
def toSentence(list_words): # Mengubah daftar kata menjadi kalimat
    sentence = ' '.join(word for word in list_words)
    return sentence


# In[9]:


slangwords = {
    "@": "di", "abis": "habis", "wtb": "beli", "masi": "masih", "wts": "jual", "wtt": "tukar",
    "bgt": "banget", "maks": "maksimal", "blm": "belum", "tdk": "tidak", "gk": "nggak", "ga": "nggak",
    "trs": "terus", "udh": "sudah", "dgn": "dengan", "tp": "tapi", "krn": "karena", "dg": "dengan",
    "sm": "sama", "jg": "juga", "lg": "lagi", "sbnrnya": "sebenarnya", "skrg": "sekarang",
    "klo": "kalau", "kl": "kalau", "dr": "dari", "jd": "jadi", "trnyata": "ternyata",
    "td": "tadi", "msh": "masih", "spt": "seperti", "sy": "saya", "gw": "saya", "gue": "saya",
    "elo": "kamu", "lu": "kamu", "lo": "kamu", "ane": "saya", "ente": "kamu", "y": "ya",
    "tdk": "tidak", "ok": "oke", "okey": "oke", "wkwk": "", "haha": "", "hehe": "", "hihi": "",
    "cm": "cuma", "aja": "saja", "ampun": "", "plz": "tolong", "brb": "sebentar", "ttyl": "nanti ngobrol lagi",
    "thx": "terima kasih", "ty": "terima kasih", "gmn": "gimana", "gimana": "bagaimana",
    "btw": "ngomong-ngomong", "fyi": "untuk informasi", "yg": "yang", "dl": "dulu",
    "bsk": "besok", "sblm": "sebelum", "pke": "pakai", "sbnr": "sebenarnya", "kmrn": "kemarin",
    "omg": "ya Tuhan", "wkwkwk": "", "cmn": "cuman", "jdwl": "jadwal", "ngga": "tidak",
    "sbnrnya": "sebenarnya", "maaciw": "terima kasih", "cuan": "keuntungan", "yey": "senang"
}

def fix_slangwords(text):
    words = text.split()
    fixed_words = []
 
    for word in words:
        if word.lower() in slangwords:
            fixed_words.append(slangwords[word.lower()])
        else:
            fixed_words.append(word)
 
    fixed_text = ' '.join(fixed_words)
    return fixed_text


# In[10]:


# Membersihkan teks dan menyimpannya di kolom 'text_clean'
clean_df['text_clean'] = clean_df['content'].apply(cleaningText)
 
# Mengubah huruf dalam teks menjadi huruf kecil dan menyimpannya di 'text_casefoldingText'
clean_df['text_casefoldingText'] = clean_df['text_clean'].apply(casefoldingText)
 
# Mengganti kata-kata slang dengan kata-kata standar dan menyimpannya di 'text_slangwords'
clean_df['text_slangwords'] = clean_df['text_casefoldingText'].apply(fix_slangwords)
 
# Memecah teks menjadi token (kata-kata) dan menyimpannya di 'text_tokenizingText'
clean_df['text_tokenizingText'] = clean_df['text_slangwords'].apply(tokenizingText)
 
# Menghapus kata-kata stop (kata-kata umum) dan menyimpannya di 'text_stopword'
clean_df['text_stopword'] = clean_df['text_tokenizingText'].apply(filteringText)
 
# Menggabungkan token-token menjadi kalimat dan menyimpannya di 'text_akhir'
clean_df['text_akhir'] = clean_df['text_stopword'].apply(toSentence)


# In[11]:


clean_df


# In[12]:


# Membaca data kamus kata-kata positif dari GitHub
lexicon_positive = dict()
 
response = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv')
# Mengirim permintaan HTTP untuk mendapatkan file CSV dari GitHub
 
if response.status_code == 200:
    # Jika permintaan berhasil
    reader = csv.reader(StringIO(response.text), delimiter=',')
    # Membaca teks respons sebagai file CSV menggunakan pembaca CSV dengan pemisah koma
 
    for row in reader:
        # Mengulangi setiap baris dalam file CSV
        lexicon_positive[row[0]] = int(row[1])
        # Menambahkan kata-kata positif dan skornya ke dalam kamus lexicon_positive
else:
    print("Failed to fetch positive lexicon data")
 
# Membaca data kamus kata-kata negatif dari GitHub
lexicon_negative = dict()
 
response = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv')
# Mengirim permintaan HTTP untuk mendapatkan file CSV dari GitHub
 
if response.status_code == 200:
    # Jika permintaan berhasil
    reader = csv.reader(StringIO(response.text), delimiter=',')
    # Membaca teks respons sebagai file CSV menggunakan pembaca CSV dengan pemisah koma
 
    for row in reader:
        # Mengulangi setiap baris dalam file CSV
        lexicon_negative[row[0]] = int(row[1])
        # Menambahkan kata-kata negatif dan skornya dalam kamus lexicon_negative
else:
    print("Failed to fetch negative lexicon data")


# In[13]:


# Fungsi untuk menentukan polaritas sentimen dari teks
def sentiment_analysis_lexicon_indonesia(text):
    score = 0
    # Inisialisasi skor sentimen ke 0

    for word in text:
        # Mengulangi setiap kata dalam teks

        if word in lexicon_positive:
            score += lexicon_positive[word]
            # Jika kata ada dalam kamus positif, tambahkan skornya ke skor sentimen

    for word in text:
        # Mengulangi setiap kata dalam teks (sekali lagi)

        if word in lexicon_negative:
            score += lexicon_negative[word]
            # Jika kata ada dalam kamus negatif, kurangkan skornya dari skor sentimen

    polarity = ''
    # Inisialisasi variabel polaritas

    # Menambahkan kategori netral
    if score > 0:
        polarity = 'positive'
        # Jika skor sentimen lebih besar dari 0, maka polaritas adalah positif
    elif score < 0:
        polarity = 'negative'
        # Jika skor sentimen kurang dari 0, maka polaritas adalah negatif
    else:
        polarity = 'neutral'
        # Jika skor sentimen sama dengan 0, maka polaritas adalah netral

    return score, polarity
    # Mengembalikan skor sentimen dan polaritas teks


# In[14]:


results = clean_df['text_stopword'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
clean_df['polarity_score'] = results[0]
clean_df['polarity'] = results[1]
print(clean_df['polarity'].value_counts())


# # **Exploratory Data Analysis (EDA)**

# #### Panjang teks

# In[15]:


# Pastikan kolom yang digunakan ada di DataFrame
df['text_length'] = df['content'].apply(lambda x: len(str(x).split()))
clean_df['clean_text_length'] = clean_df['text_akhir'].apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Histogram panjang teks sebelum pembersihan
sns.histplot(df['text_length'], bins=30, kde=True, ax=axes[0])
axes[0].set_title("Distribusi Panjang Teks Sebelum Pembersihan")

# Histogram panjang teks setelah pembersihan
sns.histplot(clean_df['clean_text_length'], bins=30, kde=True, color='green', ax=axes[1])
axes[1].set_title("Distribusi Panjang Teks Setelah Pembersihan")

plt.tight_layout()
plt.show()


# #### Kata paling sering muncul

# In[16]:


all_clean_text = " ".join(clean_df['text_akhir'])

# Hitung kata yang paling sering muncul
word_counts = Counter(all_clean_text.split())
common_words = pd.DataFrame(word_counts.most_common(20), columns=['word', 'count'])

# Plot bar chart
plt.figure(figsize=(8, 4))
sns.barplot(y=common_words['word'], x=common_words['count'], palette='coolwarm')
plt.title("Kata Paling Sering Muncul Setelah Pembersihan")
plt.xlabel("Jumlah Kemunculan")
plt.ylabel("Kata")
plt.show()


# #### Distribusi Data per kelas

# In[17]:


sentiment_counts = clean_df['polarity'].value_counts()
labels = sentiment_counts.index
sizes = sentiment_counts.values

colors = ['red', 'green', 'yellow']
explode = [0.05] * len(labels)

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, explode=explode, shadow=True)
plt.title("Distribusi Sentimen (3 Kelas)", fontsize=14)
plt.show()


# #### Word Cloud 3 Kelas

# In[18]:


clean_df['text_cleaned'] = clean_df['text_stopword'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x)).fillna("")

sentiments = ['positive', 'neutral', 'negative']

wordclouds = {}
for sentiment in sentiments:
    text = " ".join(clean_df[clean_df['polarity'] == sentiment]['text_cleaned'])
    wordclouds[sentiment] = WordCloud(width=800, height=400, background_color='white', 
                                      max_words=200, collocations=False).generate(text)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten() 

for i, sentiment in enumerate(sentiments):
    axes[i].imshow(wordclouds[sentiment], interpolation='bilinear')
    axes[i].axis("off")
    axes[i].set_title("WordCloud " + sentiment.capitalize(), fontsize=14)

if len(axes) > len(sentiments):
    for j in range(len(sentiments), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# # **4.Data Splitting, Data Preprocessing dan Ekstraksi Fitur**

# ## 4.1. Preprocessing Data

# In[19]:


# Siapkan data
texts = clean_df['text_akhir'].tolist()
labels = clean_df['polarity'].tolist()

# Encode label
le = LabelEncoder()
labels_enc = le.fit_transform(labels)
labels_cat = to_categorical(labels_enc)


# ## 4.2. Ekstraksi Fitur

# ### Word Embedding

# In[20]:


# Tokenisasi
max_words = 20000  # batas kata terbanyak yang dipakai
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding sequences
max_len = 100  # panjang maksimal tiap sequence, bisa disesuaikan
X_seq = pad_sequences(sequences, maxlen=max_len)
y = labels_cat


# ### Word2Vec

# In[21]:


tokenized_texts = [text.split() for text in texts]
embedding_dim = 128

w2v_model = Word2Vec(
    sentences=tokenized_texts,
    vector_size=embedding_dim,
    window=5,
    min_count=1,
    workers=4,
    seed=42
)

# Membuat embedding matrix untuk semua kata di tokenizer
word_index = tokenizer.word_index
embedding_matrix = np.zeros((min(max_words, len(word_index) + 1), embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        if word in w2v_model.wv:
            embedding_vector = w2v_model.wv[word]
            embedding_matrix[i] = embedding_vector


# ## 4.3. Data Splitting

# ### LSTM - Word Embedding - 70/30

# In[22]:


X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_seq, y, test_size=0.3, random_state=42, stratify=labels_enc)


# ### CNN - Word Embedding - 80/20

# In[23]:


X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_seq, y, test_size=0.2, random_state=42, stratify=labels_enc)


# ### LSTM - Word2Vec - 80/20

# In[24]:


X_train_lstmw, X_test_lstmw, y_train_lstmw, y_test_lstmw = train_test_split(
    X_seq, y, test_size=0.2, random_state=42, stratify=labels_enc)


# # **5. Membangun Model**

# In[25]:


model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model_lstm.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(3, activation='softmax'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.build(input_shape=(None, max_len))
model_lstm.summary()


# In[26]:


model_cnn = Sequential()
model_cnn.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model_cnn.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(64, activation='relu'))
model_cnn.add(Dropout(0.2))
model_cnn.add(Dense(3, activation='softmax'))
model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn.build(input_shape=(None, max_len))
model_cnn.summary()


# In[27]:


model_lstmw = Sequential()
model_lstmw.add(Embedding(input_dim=embedding_matrix.shape[0],
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=True))
model_lstmw.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model_lstmw.add(Dropout(0.2))
model_lstmw.add(Dense(3, activation='softmax'))
model_lstmw.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstmw.build(input_shape=(None, max_len))
model_lstmw.summary()


# # **5. Latih Model dan Early Stoping**

# ## 5.1. Early Stoping

# In[28]:


early_stop = EarlyStopping(monitor='val_loss', patience=3)


# ## 5.2. Melatih Model

# ### LSTM Word Embedding

# In[29]:


history = model_lstm.fit(X_train_lstm, y_train_lstm, 
                    epochs=10, 
                    batch_size=128, 
                    validation_split=0.1, 
                    callbacks=[early_stop])


# ### CNN Word Embedding

# In[30]:


history = model_cnn.fit(X_train_cnn, y_train_cnn, 
                    epochs=10, 
                    batch_size=128, 
                    validation_split=0.1, 
                    callbacks=[early_stop])


# ### LSTM Word2Vec

# In[31]:


history = model_lstmw.fit(X_train_lstmw, y_train_lstmw, 
                    epochs=10, 
                    batch_size=128, 
                    validation_split=0.1, 
                    callbacks=[early_stop])


# # **6. Evaluasi Model**

# ### LSTM  Word Embedding

# In[32]:


loss_train_lstm, accuracy_train_lstm = model_lstm.evaluate(X_train_lstm, y_train_lstm)
loss_test_lstm, accuracy_test_lstm = model_lstm.evaluate(X_test_lstm, y_test_lstm)
print(f'Train Accuracy LSTM: {accuracy_train_lstm:.2f} %')
print(f'Test Accuracy LSTM: {accuracy_test_lstm:.2f} %')

y_pred_prob_lstm = model_lstm.predict(X_test_lstm)
y_pred_lstm = np.argmax(y_pred_prob_lstm, axis=1)
y_true_lstm = np.argmax(y_test_lstm, axis=1)

print("\nClassification Report LSTM:")
print(classification_report(y_true_lstm, y_pred_lstm, target_names=le.classes_))

cm = confusion_matrix(y_true_lstm, y_pred_lstm)
print("Confusion Matrix LSTM:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix LSTM")
plt.show()


# In[33]:


loss_train_cnn, accuracy_train_cnn = model_cnn.evaluate(X_train_cnn, y_train_cnn)
loss_test_cnn, accuracy_test_cnn = model_cnn.evaluate(X_test_cnn, y_test_cnn)
print(f'Train Accuracy CNN: {accuracy_train_cnn:.2f} %')
print(f'Test Accuracy CNN: {accuracy_test_cnn:.2f} %')

y_pred_prob_cnn = model_cnn.predict(X_test_cnn)
y_pred_cnn = np.argmax(y_pred_prob_cnn, axis=1)
y_true_cnn = np.argmax(y_test_cnn, axis=1)

print("\nClassification Report CNN:")
print(classification_report(y_true_cnn, y_pred_cnn, target_names=le.classes_))

cm = confusion_matrix(y_true_cnn, y_pred_cnn)
print("Confusion Matrix CNN:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix CNN")
plt.show()


# In[34]:


loss_train_lstmw, accuracy_train_lstmw = model_lstmw.evaluate(X_train_lstmw, y_train_lstmw)
loss_test_lstmw, accuracy_test_lstmw = model_lstmw.evaluate(X_test_lstmw, y_test_lstmw)
print(f'Train Accuracy LSTMW: {accuracy_train_lstmw:.2f} %')
print(f'Test Accuracy LSTMW: {accuracy_test_lstmw:.2f} %')

y_pred_prob_lstmw = model_lstmw.predict(X_test_lstmw)
y_pred_lstmw = np.argmax(y_pred_prob_lstmw, axis=1)
y_true_lstmw = np.argmax(y_test_lstmw, axis=1)

print("\nClassification Report LSTMW:")
print(classification_report(y_true_lstmw, y_pred_lstmw, target_names=le.classes_))

cm = confusion_matrix(y_true_lstmw, y_pred_lstmw)
print("Confusion Matrix LSTMW:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix LSTMW")
plt.show()


# ## **7. Inference (Testing)**

# In[35]:


text_input = "benerin tuh jaringan ngeleg bett mahal doang"

def preprocess_text(text, tokenizer, max_len=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    return padded

processed_text = preprocess_text(text_input, tokenizer, max_len)

pred_lstm   = model_lstm.predict(processed_text)
pred_cnn    = model_cnn.predict(processed_text)
pred_lstmw  = model_lstmw.predict(processed_text)

class_lstm   = np.argmax(pred_lstm, axis=1)
class_cnn    = np.argmax(pred_cnn, axis=1)
class_lstmw  = np.argmax(pred_lstmw, axis=1)

sentiment_lstm   = le.inverse_transform(class_lstm)[0]
sentiment_cnn    = le.inverse_transform(class_cnn)[0]
sentiment_lstmw  = le.inverse_transform(class_lstmw)[0]

print(f"Model LSTM: {sentiment_lstm}")
print(f"Model CNN: {sentiment_cnn}")
print(f"Model LSTM Word2Vec: {sentiment_lstmw}")

votes = [sentiment_lstm, sentiment_cnn, sentiment_lstmw]
final_sentiment = max(set(votes), key=votes.count)
print(f"Final Sentimen (Voting): {final_sentiment}")


# In[37]:


print("Index for 'lemot':", tokenizer.word_index.get("lemot"))

