import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

sns.set(style='darkgrid')

# Load DataFrame
df = pd.read_csv("main_data.csv")

# Konversi kolom tanggal
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

# Sidebar dengan filter tanggal
min_date = df['datetime'].min()
max_date = df['datetime'].max()

with st.sidebar:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=min_date, max_value=max_date, value=[min_date, max_date]
    )

# Filter DataFrame berdasarkan rentang waktu
main_df = df[(df["datetime"] >= str(start_date)) & (df["datetime"] <= str(end_date))]

# 1. Tren harian rata-rata polutan
st.subheader("Tren Harian Kualitas Udara")
daily_trend = main_df.groupby('datetime')[['PM2.5', 'PM10']].mean().reset_index()
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(data=daily_trend, x='datetime', y='PM2.5', label='PM2.5', marker='o', ax=ax)
sns.lineplot(data=daily_trend, x='datetime', y='PM10', label='PM10', marker='o', ax=ax)
ax.set_xlabel("Tanggal")
ax.set_ylabel("Konsentrasi")
st.pyplot(fig)

# 2. Distribusi PM2.5 dan PM10
st.subheader("Distribusi Polutan PM2.5 dan PM10")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(main_df['PM2.5'], kde=True, bins=30, color='skyblue', ax=axes[0])
axes[0].set_title("Distribusi PM2.5")
sns.histplot(main_df['PM10'], kde=True, bins=30, color='salmon', ax=axes[1])
axes[1].set_title("Distribusi PM10")
st.pyplot(fig)

# 3. Pengaruh suhu, kelembapan, dan cuaca terhadap polutan
st.subheader("Hubungan Suhu, Kelembapan, dan Cuaca terhadap Polutan")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
sns.regplot(x='TEMP', y='PM2.5', data=main_df, ax=axes[0, 0])
axes[0, 0].set_title("Temperature vs PM2.5")
sns.regplot(x='TEMP', y='PM10', data=main_df, ax=axes[1, 0])
axes[1, 0].set_title("Temperature vs PM10")
sns.regplot(x='DEWP', y='PM2.5', data=main_df, ax=axes[0, 1])
axes[0, 1].set_title("Dew Point vs PM2.5")
sns.regplot(x='DEWP', y='PM10', data=main_df, ax=axes[1, 1])
axes[1, 1].set_title("Dew Point vs PM10")
sns.boxplot(x='wd', y='PM2.5', data=main_df, ax=axes[0, 2])
axes[0, 2].set_title("Wind Direction vs PM2.5")
sns.boxplot(x='wd', y='PM10', data=main_df, ax=axes[1, 2])
axes[1, 2].set_title("Wind Direction vs PM10")
st.pyplot(fig)

# 4. Perbedaan kualitas udara antar stasiun
st.subheader("Perbandingan Kualitas Udara Antar Stasiun")
df_station = main_df.groupby('station')[['PM2.5', 'PM10']].mean().reset_index()
fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(data=df_station, x='station', y='PM2.5', color='skyblue', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# 5. Perbedaan kualitas udara hari kerja vs hari libur
st.subheader("Perbandingan Kualitas Udara Hari Kerja vs Hari Libur")
main_df['weekday'] = main_df['datetime'].dt.weekday
main_df['day_type'] = main_df['weekday'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='day_type', y='PM2.5', data=main_df, palette='Reds', ax=ax)
st.pyplot(fig)

# 6. Pola pencemaran udara dalam sehari
st.subheader("Pola Pencemaran Udara dalam Sehari")
main_df['hour'] = main_df['datetime'].dt.hour
df_hourly = main_df.groupby('hour')[['PM2.5', 'PM10']].mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df_hourly, x='hour', y='PM2.5', marker='o', label='PM2.5', ax=ax)
sns.lineplot(data=df_hourly, x='hour', y='PM10', marker='o', label='PM10', ax=ax)
st.pyplot(fig)

# 7. Stasiun dengan kualitas udara terburuk
st.subheader("Stasiun dengan Kualitas Udara Terburuk")
df_station_pm25 = main_df.groupby('station')[['PM2.5']].mean().reset_index()
worst_station = df_station_pm25.sort_values(by='PM2.5', ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=worst_station, x='station', y='PM2.5', color='green', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# 8. Waktu terbaik untuk aktivitas di luar rumah
st.subheader("Waktu Terbaik untuk Aktivitas Luar Ruangan")
best_hour = df_hourly.loc[df_hourly['PM2.5'].idxmin(), 'hour']
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df_hourly, x='hour', y='PM2.5', marker='o', label='PM2.5', ax=ax)
ax.axvline(x=best_hour, color='green', linestyle='--', label=f'Waktu Terbaik: {best_hour} Jam')
st.pyplot(fig)

# 9. Hari dengan kualitas udara terburuk dalam seminggu
st.subheader("Hari dengan Kualitas Udara Terburuk dalam Seminggu")
main_df['day_name'] = main_df['datetime'].dt.day_name()
df_day = main_df.groupby('day_name')['PM2.5'].mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=df_day, x='day_name', y='PM2.5', color='brown', ax=ax)
st.pyplot(fig)
