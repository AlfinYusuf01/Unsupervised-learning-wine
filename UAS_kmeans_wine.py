import streamlit as st 
import pandas as pd 
import pickle
import numpy as np

# Memuat model
with open('wine-clustering.pkl', 'rb') as f: 
    model_kmeans = pickle.load(f)

st.title("Prediksi Kelompok Anggur Menggunakan KMeans") 
st.markdown("Output utama dari model ini adalah pengelompokan data ke dalam klaster-klaster berdasarkan fitur yang diberikan.")
st.sidebar.title("Inputkan data Anda di sini")

# meload model
st.sidebar.text('Model untuk Prediksi: Kmeans') 
model = model_kmeans

# Inisialisasi atau reset hasil jika model berubah
Alcohol = st.sidebar.slider("Kandungan Alkohol (%):", 0, 100, 0)
Malic_Acid = st.sidebar.number_input('Kandungan Asam Malat (Malic Acid):', min_value=0)
Ash = st.sidebar.number_input('Kandungan Ash (Abu):', min_value=0)
Ash_Alcanity = st.sidebar.number_input('Kandungan Ash Alkalinity:', min_value=0)
Magnesium = st.sidebar.number_input('Kandungan Magnesium:', min_value=0)
Total_Phenols = st.sidebar.number_input('Kandungan Total Fenol:', min_value=0)
Flavanoids = st.sidebar.number_input('Kandungan Flavanoid:', min_value=0)
Nonflavanoid_Phenols = st.sidebar.number_input('Kandungan Non-Flavanoid Fenol:', min_value=0)
Proanthocyanins = st.sidebar.number_input('Kandungan Proanthocyanins:', min_value=0)
Color_Intensity = st.sidebar.number_input('Intensitas Warna (Color Intensity):', min_value=0)
Hue = st.sidebar.number_input('Hue (Warna):', min_value=0)
OD280 = st.sidebar.number_input('OD280/OD315 Rasio (Absorbansi):', min_value=0)
Proline = st.sidebar.number_input('Kandungan Proline:', min_value=0)

# Membuat array untuk fitur input pengguna
features = np.array([Alcohol, Malic_Acid, Ash, Ash_Alcanity, Magnesium, Total_Phenols, Flavanoids,
                     Nonflavanoid_Phenols, Proanthocyanins, Color_Intensity, Hue, OD280, Proline])

col1, col2 = st.columns([2,2])
with col1:
    # Tombol untuk mengkelompokkan kelas
    if st.sidebar.button('Klasterkan Data'):
        try:
            predicted_klaster = model.predict(features.reshape(1, -1))[0]
            predicted_klaster_label = ['Type-1', 'Type-2', 'Type-3'][predicted_klaster]
            st.success(f"Data telah berhasil dikelompokkan dalam klaster: {predicted_klaster_label}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memprediksi: {e}")

        # Menyimpan hasil ke dalam session_state jika belum ada
        if 'results' not in st.session_state:
            st.session_state['results'] = []
        
        # Menyimpan hasil ke dalam session_state
        st.session_state['results'].append({
            'Alcohol': Alcohol,
            'Malic_Acid': Malic_Acid,
            'Ash': Ash,
            'Ash_Alcanity': Ash_Alcanity,
            'Magnesium': Magnesium,
            'Total_Phenols': Total_Phenols,
            'Flavanoids': Flavanoids,
            'Nonflavanoid_Phenols': Nonflavanoid_Phenols,
            'Proanthocyanins': Proanthocyanins,
            'Color_Intensity': Color_Intensity,
            'Hue': Hue,
            'OD280': OD280,
            'Proline': Proline, 
            'predicted_cluster': predicted_klaster_label
        })
        
        # Batasi jumlah hasil yang disimpan (misalnya hanya 10 hasil terakhir)
        if len(st.session_state['results']) > 10:
            st.session_state['results'] = st.session_state['results'][-10:]
with col2:
    if st.sidebar.button('Reset Hasil'):
        st.session_state['reset'] = True
    
    if 'reset' in st.session_state and st.session_state['reset']:
        if 'results' in st.session_state:
            del st.session_state['results']
        st.session_state['reset'] = False
        st.success('Hasil telah direset!')
        st.experimental_rerun()


# Menampilkan semua hasil prediksi dalam tabel
if 'results' in st.session_state and st.session_state['results']:
    result_df = pd.DataFrame(st.session_state['results'])
    st.subheader('Tabel Hasil Prediksi')
    st.dataframe(result_df)
