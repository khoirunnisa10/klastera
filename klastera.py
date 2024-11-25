import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from io import StringIO
import time

# Inisialisasi variabel global untuk model dan prediksi
model = None
X_train, X_test, y_train, y_test, y_pred = None, None, None, None, None

# Fungsi untuk halaman Home
def home_page():
    st.title("Hitung Klasifikasi bersama Klastera")
    st.markdown(
        """<div style='text-align: justify;'>
        Klasifikasi adalah metode dalam machine learning yang bertujuan untuk mengelompokkan data ke dalam kategori atau kelas tertentu.
        Pada dasarnya, klasifikasi mencoba memprediksi label atau kelas dari data yang belum diketahui berdasarkan pola yang dipelajari dari data yang sudah diketahui.
        </div>""", unsafe_allow_html=True
    )

    # Sidebar pengaturan
    uploaded_file = st.sidebar.file_uploader("Unggah file CSV/XLSX", type=["csv", "xlsx"])
    
    global model, X_train, X_test, y_train, y_test, y_pred

    if uploaded_file is not None:
        # Baca file yang diunggah
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Opsi drop column
        st.sidebar.subheader("Kolom yang akan dihapus")
        drop_columns = st.sidebar.multiselect("Pilih Kolom untuk Dihapus", data.columns)
        if drop_columns:
            data = data.drop(columns=drop_columns)

        # Pilih kolom target
        st.sidebar.subheader("Parameter Klasifikasi")
        target_column = st.sidebar.selectbox("Pilih Kolom Label (nilai Y)", data.columns)
        test_size = st.sidebar.slider("Ukuran Data Uji (%)", 0.1, 0.9, 0.2)
        algorithm = st.sidebar.selectbox(
            "Pilih Algoritma Klasifikasi", 
            ["Logistic Regression", "Random Forest", "Naive Bayes", "SVM", "KNN", "Decision Tree", "ANN"]
        )
        k_neighbors = st.sidebar.slider("Jumlah K (hanya untuk KNN)", 1, 20, 5)
        
        # Pilihan kolom yang akan dienkode
        encode_columns = st.sidebar.multiselect("Pilih Kolom untuk Encoding", 
                                                [col for col in data.columns if col != target_column])

        # Tampilkan data yang diunggah
        st.header("Data yang Diupload")
        st.write(data)
        
        # Transformasi data karakter ke numerik
        for column in encode_columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))

        # Membagi data menjadi data pelatihan dan pengujian
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Tampilkan pembagian data
        st.subheader("Pembagian Data")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Pelatihan**:", X_train.shape[0], "baris")
            st.write("**Data Pengujian**:", X_test.shape[0], "baris")

        # Model klasifikasi
        if st.sidebar.button("Hitung Model"):
            if algorithm == "Logistic Regression":
                model = LogisticRegression(max_iter=200)
            elif algorithm == "Random Forest":
                model = RandomForestClassifier()
            elif algorithm == "Naive Bayes":
                model = GaussianNB()
            elif algorithm == "SVM":
                model = SVC()
            elif algorithm == "KNN":
                model = KNeighborsClassifier(n_neighbors=k_neighbors)
            elif algorithm == "Decision Tree":
                model = DecisionTreeClassifier()
            elif algorithm == "ANN":
                model = MLPClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.success("Model berhasil dihitung!")

# Fungsi untuk halaman Lihat Akurasi
def lihat_akurasi_page():
    # st.title("Lihat Akurasi Model")
    if model is not None and y_test is not None and y_pred is not None:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Menggunakan kolom untuk hasil metrik utama
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Akurasi", f"{accuracy:.2f}")
        with col2:
            precision = precision_score(y_test, y_pred, average='weighted')
            st.metric("Precision", f"{precision:.2f}")
        with col3:
            recall = recall_score(y_test, y_pred, average='weighted')
            st.metric("Recall", f"{recall:.2f}")
        with col4:
            f1 = f1_score(y_test, y_pred, average='weighted')
            st.metric("F1 Score", f"{f1:.2f}")

        # Visualisasi Confusion Matrix
        st.subheader("📋 Matriks Kesalahan (Confusion Matrix)")
        cn = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cn, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 8})
        ax.set_xlabel("Prediksi", fontsize=10)
        ax.set_ylabel("Asli", fontsize=10)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # Analisis Confusion Matrix
        st.subheader("Analisis Confusion Matrix")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**True Positive (TP)**:", cn[1, 1])
            st.write("Prediksi benar untuk kelas positif.")
            st.write("**True Negative (TN)**:", cn[0, 0])
            st.write("Prediksi benar untuk kelas negatif.")                    
        with col2:
            st.write("**False Positive (FP)**:", cn[0, 1])
            st.write("Prediksi salah untuk kelas positif.")
            st.write("**False Negative (FN)**:", cn[1, 0])
            st.write("Prediksi salah untuk kelas negatif.")
         
         # Tambahkan kesimpulan akhir
        # Penjelasan sederhana
        st.markdown("""
        <div style="text-align: justify;">
        Matriks kesalahan membantu memahami bagaimana model melakukan prediksi:
        <ul>
            <li><strong>Diagonal biru</strong>: Jumlah prediksi yang benar.</li>
            <li><strong>Di luar diagonal</strong>: Prediksi salah (error).</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Hitung model terlebih dahulu di halaman Home.")
    st.warning("Hitung model terlebih dahulu di halaman Home.")

def lihat_klasifikasi_page():
    if y_pred is not None and y_test is not None:
        st.subheader("📈 Hasil Klasifikasi Model")
        
        # Ringkasan hasil prediksi
        total_data = len(y_test)
        total_correct = sum(y_test == y_pred)
        total_error = total_data - total_correct
        accuracy = (total_correct / total_data) * 100

        st.markdown(f"""
        <div style="text-align: justify;">
            Dari total <strong>{total_data}</strong> data uji:
            <ul>
                <li><strong>{total_correct}</strong> prediksi benar</li>
                <li><strong>{total_error}</strong> prediksi salah</li>
            </ul>
            Akurasi model adalah <strong>{accuracy:.2f}%</strong>.
        </div>
        """, unsafe_allow_html=True)

        # Visualisasi distribusi kelas
        st.subheader("📊 Distribusi Kelas")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Visualisasi data
        sns.histplot(y_train, label="Data Latih", color='skyblue', kde=False, stat="count", ax=ax, binwidth=0.5)
        sns.histplot(y_test, label="Data Uji (Asli)", color='orange', kde=False, stat="count", ax=ax, binwidth=0.5)
        sns.histplot(y_pred, label="Hasil Prediksi", color='green', kde=False, stat="count", ax=ax, binwidth=0.5)

        # Menambahkan elemen visual yang membantu
        ax.set_title("Distribusi Data Latih, Uji, dan Prediksi", fontsize=16, fontweight='bold')
        ax.set_xlabel("Kategori Kelas", fontsize=14)
        ax.set_ylabel("Jumlah Observasi", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(title="Legenda", fontsize=12, title_fontsize=14, loc='upper right')

        # Menampilkan grafik di Streamlit
        st.pyplot(fig)

        # Tabel perbandingan hasil prediksi
        st.subheader("🔍 Perbandingan Prediksi vs Data Asli")
        comparison_df = pd.DataFrame({"Asli (Data Uji)": y_test, "Prediksi": y_pred})
        st.write("Berikut adalah perbandingan 10 data pertama:")
        st.dataframe(comparison_df.head(10))
    else:
        st.warning("⚠️ Silakan hitung model terlebih dahulu di halaman Home.")


# Main aplikasi
st.sidebar.markdown("<h1 style='font-size: 40px;'>Klastera</h1>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🏠Home", "📊Lihat Akurasi", "📈Lihat Klasifikasi"])

with tab1:
    home_page()
with tab2:
    lihat_akurasi_page()
with tab3:
    lihat_klasifikasi_page()
