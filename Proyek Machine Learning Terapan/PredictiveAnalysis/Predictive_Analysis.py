#!/usr/bin/env python
# coding: utf-8

# ## **1. Import Library**

# Mengimport library-library yang dibutuhkan untuk membuat model

# In[1]:


import numpy as np # aljabar linear dan manipulasi angka
import matplotlib.pyplot as plt # visualisasi
import pandas as pd # manipulasi data
import seaborn as sns # visualisasi

from sklearn.preprocessing import OneHotEncoder, StandardScaler # encoding fitur kategori dan standarisasi fitur numerik
from sklearn.model_selection import train_test_split # membagi dataset ke train dan test
from sklearn.ensemble import RandomForestClassifier # algoritma randomforest
from xgboost import XGBClassifier # algoritma xgboost
from lightgbm import LGBMClassifier # algoritma lightgbm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # evaluasi model


# ## **2. Data Loading**

# Load dataset dari path

# In[2]:


path = "/mnt/d/Artificial Intellegence/Machine Learning/CodingCamp/Proyek Machine Learning Terapan/PredictiveAnalysis/diabetes_prediction_dataset.csv"
df = pd.read_csv(path)


# Melihat sample dari data

# In[3]:


df.head()


# ## **3. Exploratory Data Analysis (EDA)**

# ### 3.1. Deskripsi Variabel

# Malihat struktur dataset, dari hasil nampaknya tidak ada missing value.

# In[4]:


df.info()


# Analisis statistik deskriptif pada variabel numerik, ada indikasi outlier fitur blood_glucose_level, HbA1c_level dan bmi.

# In[5]:


df.describe()


# ### 3.2. Menangani Missing Value dan Outlier

# Mengecek dataset apakah terdapat missing values

# In[6]:


df.isnull().sum()


# Mengecek total kategori pada fitur gender, ada kategori yang hanya ada 18 dari total keseluruhan data dan sangat timpang jumlah data keseluruhan.

# In[7]:


print(df['gender'].value_counts())


# Mangganti kategori yang jumlahnya sekitar 0.001% dengan modus dari kategori lain.

# In[8]:


gender_mode = df['gender'].mode()[0]  

df['gender'] = df['gender'].replace('Other', gender_mode)


# Mengecek dan menangani Outlier

# In[9]:


numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

outlierValues = {}

fig, axes = plt.subplots(1, len(numerical_features), figsize=(5 * len(numerical_features), 5))

if len(numerical_features) == 1:
    axes = [axes]

for ax, col in zip(axes, numerical_features):
    sns.boxplot(x=df[col], ax=ax)
    ax.set_title(f"Before: {col}")

plt.tight_layout()
plt.show()

for col in numerical_features:
    q25 = np.percentile(df[col].dropna(), 25)
    q75 = np.percentile(df[col].dropna(), 75)
    iqr = q75 - q25
    lowerBound = q25 - 1.5 * iqr
    upperBound = q75 + 1.5 * iqr

    outliers = df[col][(df[col] < lowerBound) | (df[col] > upperBound)]
    outlierValues[col] = outliers

    df[col] = np.clip(df[col], lowerBound, upperBound)

print("\nJumlah Outlier per Kolom (Sebelum Clipping):")
for col, outliers in outlierValues.items():
    print(f"{col}: {len(outliers)} outlier")

fig, axes = plt.subplots(1, len(numerical_features), figsize=(5 * len(numerical_features), 5))

if len(numerical_features) == 1:
    axes = [axes]

for ax, col in zip(axes, numerical_features):
    sns.boxplot(x=df[col], ax=ax)
    ax.set_title(f"After: {col}")

plt.tight_layout()
plt.show()


# ### 3.3. Univariate Analysis

# Mengelompokan kolom ke numerik dan kategorikal.

# In[10]:


numerical_features = ['age','bmi','HbA1c_level','blood_glucose_level']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'smoking_history', 'diabetes']


# Memvisualisasi untuk melihat distibusi nilai dari semua kolom numerik.

# In[40]:


plt.figure(figsize=(5 * len(numerical_features), 4))  

for i, col in enumerate(numerical_features, 1):
    plt.subplot(1, len(numerical_features), i)  
    
    color = sns.color_palette("husl", len(numerical_features))[i-1]  
    
    if df[col].nunique() > 10:
        sns.histplot(df[col].dropna(), bins=30, kde=True, color=color)
    else:
        sns.countplot(x=df[col], palette="husl")

    plt.title(f"Distribusi {col}")
    plt.xlabel(col)
    plt.ylabel("Frekuensi")

plt.tight_layout()  # Menghindari tumpang tindih
plt.show()


# Memvisualisasi untuk melihat distibusi persebaran nilai dari masing masing kategori dari semua kolom Kategori.

# In[12]:


plt.figure(figsize=(15, 13))

palettes = [
    "viridis", "magma", "coolwarm", "plasma", "cividis", "inferno",
]

for col in categorical_features:
    count = df[col].value_counts()
    percent = 100 * df[col].value_counts(normalize=True)
    
    df_count = pd.DataFrame({'Jumlah Sampel': count, 'Persentase': percent.round(1)})
    
    print(f"\nDistribusi Kategori untuk '{col}':")
    print(df_count)
    print("-" * 50)

for i, (col, palette) in enumerate(zip(categorical_features, palettes), 1):
    plt.subplot(4, 3, i)
    sns.countplot(y=df[col], order=df[col].value_counts().index, hue=df[col], palette=palette, legend=False)
    plt.title(f"Distribusi {col}")
    plt.xlabel("Frekuensi")
    plt.ylabel(col)

plt.tight_layout()
plt.show()


# ### 3.4. Multivariate Analysis

# Melihat rata rata diabetes vs fitur lain dalam dataset

# In[13]:


categorical_features = ['gender', 'hypertension', 'heart_disease', 'smoking_history']

fig, axes = plt.subplots(1, len(categorical_features), figsize=(20, 5))

for i, col in enumerate(categorical_features):
    sns.barplot(x=col, y="diabetes", data=df, ax=axes[i], hue=col, estimator=lambda x: sum(x)/len(x))
    axes[i].set_title(f"Rata-rata 'diabetes' vs {col}")

plt.tight_layout()
plt.show()


# Memvisualisasikan hubungan antar variabel dalam dataset.

# In[14]:


sns.pairplot(df, diag_kind = 'kde')


# Visualisasi correlation matrix untuk melihat korelasi atau hubungan fitur-fitur yang ada terhadap target atau label (diabetes).

# In[15]:


numerical_features = df.select_dtypes(include=['number']).columns

plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_features].corr().round(2)
 
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)


# ## **4. Data Preparation**

# ### 4.1. Encoding Fitur Kategori

# Mengencoding fitur kategori (gender dan smoking_history) ke fitur numerik menggunakan OneHotEncoder.

# In[ ]:


categorical_features = ['gender', 'smoking_history']
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

encoded_array = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_features), index=df.index)

df = df.drop(columns=categorical_features)
df = pd.concat([df, encoded_df], axis=1)


# Mengecek hasil dari OneHotEncoder yang membuat fitur baru dari hasil encoding.

# In[17]:


df.head()


# ### 4.2. Standarisasi Fitur

# Melakukan standarisasi atau normalisasi fitur numerik yang rentangnya beragam seperti kolom 'age', 'bmi', 'blood_glucose_level' dan 'HbA1c_level' menggunakan StandardScaler untuk menyamakan rentang nilai.

# In[18]:


numerical_features = ['age', 'bmi', 'blood_glucose_level', 'HbA1c_level']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])


# Melihat hasil perubahan dari nilai pada fitur numerik yang ditentukan sebelumnya.

# In[19]:


df.head()


# ### 4.3. Train-Test-Split

# Membagi data menjadi train dan test, 70% train atau 70000 sampel dan 30% atau 30000 untuk test.

# In[20]:


X = df.drop(columns=["diabetes"])  
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# ## **5. Model Development**

# ### 5.1. Random Forest

# Load model Random Forest dan melatihnya dengan data train.

# In[21]:


model_randomforest = RandomForestClassifier(n_estimators=100, random_state=123)
model_randomforest.fit(X_train, y_train)


# ### 5.2. XGBoost

# Load model XGBoost dan melatihnya dengan data train.

# In[22]:


model_xgboost = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=123)
model_xgboost.fit(X_train, y_train)


# ### 5.3 LightGBM

# Load model LightGBM dan melatihnya dengan data train.

# In[23]:


model_lightgbm = LGBMClassifier(random_state=123)
model_lightgbm.fit(X_train, y_train)


# ## **6. Evaluasi Model**

# ### 6.1. Random Forest

# Evaluasi metrik accuracy_score dan classification_report (precision, recall dan f1-score) pada model Random Forest.

# In[42]:


y_test_pred = model_randomforest.predict(X_test)

test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nModel: Random Forest")
print(f"Test Accuracy: {test_acc:.4f}")

print("\n--- Classification Report Random Forest ---\n", classification_report(y_test, y_test_pred))

test_cm = confusion_matrix(y_test, y_test_pred)

print('Confusion Matrix Random Forest')
print(test_cm)

fig, axes = plt.subplots(1, figsize=(5, 4))

sns.heatmap(test_cm, annot=True, fmt='d', cmap='Greens', ax=axes)
axes.set_title("Confusion Matrix Random Forest")
axes.set_xlabel("Predicted Label")
axes.set_ylabel("True Label")

plt.tight_layout()
plt.show()


# ### 6.2. XGBoost

# Evaluasi metrik accuracy_score dan classification_report (precision, recall dan f1-score) pada model XGBoost.

# In[43]:


y_test_pred = model_xgboost.predict(X_test)

test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nModel: XGBoost")
print(f"Test Accuracy: {test_acc:.4f}")

print("\n--- Classification Report XGBoost ---\n", classification_report(y_test, y_test_pred))

test_cm = confusion_matrix(y_test, y_test_pred)

print('Confusion Matrix XGBoost')
print(test_cm)

fig, axes = plt.subplots(1, figsize=(5, 4))

sns.heatmap(test_cm, annot=True, fmt='d', cmap='Reds', ax=axes)
axes.set_title("Confusion Matrix XGBoost")
axes.set_xlabel("Predicted Label")
axes.set_ylabel("True Label")

plt.tight_layout()
plt.show()


# ### 6.3. LightGBM

# Evaluasi metrik accuracy_score dan classification_report (precision, recall dan f1-score) pada model LightGBM.

# In[45]:


y_test_pred = model_lightgbm.predict(X_test)

test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nModel: LightGBM")
print(f"Test Accuracy: {test_acc:.4f}")

print("\n--- Classification Report LightGBM ---\n", classification_report(y_test, y_test_pred))

test_cm = confusion_matrix(y_test, y_test_pred)

print('Confusion Matrix LightGBM')
print(test_cm)

fig, axes = plt.subplots(1, figsize=(5, 4))

sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=axes)
axes.set_title("Confusion Matrix LightGBM")
axes.set_xlabel("Predicted Label")
axes.set_ylabel("True Label")

plt.tight_layout()
plt.show()

