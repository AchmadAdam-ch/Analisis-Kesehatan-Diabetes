import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# --- BAGIAN 1: PENGATURAN TAMPILAN ---
st.set_page_config(page_title="Prediksi Diabetes", layout="wide")
st.title("Aplikasi Analisis & Prediksi Diabetes 🩺")
st.write("Aplikasi ini memprediksi risiko diabetes menggunakan algoritma Decision Tree.")

# --- BAGIAN 2: MEMPROSES DATA ---
# Kita gunakan fungsi @st.cache_data agar aplikasi tidak berat saat dijalankan ulang
@st.cache_data
def load_and_clean_data():
    # 1. Membaca file CSV yang Anda unggah
    df = pd.read_csv('diabetes.csv')
    
    # 2. Membersihkan data (mengganti nilai 0 dengan median)
    kolom_tidak_boleh_nol = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for kolom in kolom_tidak_boleh_nol:
        df[kolom] = df[kolom].replace(0, df[kolom].median())
    return df

df = load_and_clean_data()

# --- BAGIAN 3: MELATIH MODEL (OTAK AI) ---
# Memisahkan fitur (input) dan target (hasil)
X = df.drop('Outcome', axis=1) # Semua kolom kecuali Outcome
y = df['Outcome']              # Kolom Outcome saja

# Melatih model Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X, y)

# --- BAGIAN 4: MEMBUAT FORM INPUT USER ---
st.sidebar.header("Masukkan Data Medis Anda:")

def user_input_features():
    pregnancies = st.sidebar.number_input("Jumlah Kehamilan", 0, 20, 1)
    glucose = st.sidebar.slider("Kadar Glukosa (Gula Darah)", 0, 200, 100)
    blood_pressure = st.sidebar.slider("Tekanan Darah", 0, 130, 70)
    skin_thickness = st.sidebar.slider("Ketebalan Kulit (mm)", 0, 99, 20)
    insulin = st.sidebar.slider("Kadar Insulin", 0, 850, 30)
    bmi = st.sidebar.number_input("Indeks Massa Tubuh (BMI)", 0.0, 70.0, 25.0)
    dpf = st.sidebar.number_input("Riwayat Keturunan (Pedigree Function)", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Usia", 1, 100, 30)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- BAGIAN 5: MENAMPILKAN HASIL PREDIKSI ---
st.subheader("Data yang Anda Masukkan:")
st.write(input_df)

if st.button("Analisis Sekarang"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader("Hasil Analisis:")
    if prediction[0] == 1:
        st.error(f"⚠️ Hasil: Terdeteksi Risiko Diabetes (Keyakinan: {prediction_proba[0][1]*100:.2f}%)")
        st.write("Saran: Segera konsultasikan dengan tenaga medis untuk pemeriksaan lebih lanjut.")
    else:
        st.success(f"✅ Hasil: Tidak Terdeteksi Risiko Diabetes (Keyakinan: {prediction_proba[0][0]*100:.2f}%)")
        st.write("Saran: Tetap jaga pola makan dan olahraga rutin!")