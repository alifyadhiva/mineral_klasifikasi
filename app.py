import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import os
import tempfile
import shutil

# Coba Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

st.set_page_config(page_title="Pengenalan Gambar Mineral")

# Periksa apakah library YOLO tersedia
def cek_library():
    if not YOLO_AVAILABLE:
        st.error('Ultralytics tidak terpasang. Silakan instal dengan perintah berikut:')
        st.code('pip install ultralytics')
        return False
    return True

st.markdown("""
<div style="background-color:#0984e3; padding: 20px; text-align: center;">
<h1 style="color: white;"> Program Pengenalan Gambar </h1>
<h5 style="color: white;">Deteksi Gambar Mineral</h5>
</div>
""", unsafe_allow_html=True)

# Pastikan library sudah terpasang sebelum melanjutkan
if cek_library():
    uploaded_file = st.file_uploader("Upload gambar mineral", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        # Penyimpanan sementara
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "gambar.jpg")

        # Buka dan ubah ukuran gambar
        image = Image.open(uploaded_file)
        image = image.resize((300, 300))
        image.save(temp_file)

        # Tampilkan Gambar
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(image, caption="Gambar yang diupload")
        st.markdown("</div>", unsafe_allow_html=True)

        # Deteksi Gambar
        if st.button("Deteksi Gambar"):
            with st.spinner("Sedang Diproses..."):
                try:
                    model = YOLO("best.pt")
                    hasil = model(temp_file)

                    # Ambil hasil prediksi
                    if len(hasil) > 0 and hasil[0].probs is not None:
                        class_names = model.names  # Ambil nama kelas dari model
                        nilai_prediksi = hasil[0].probs.data.numpy().tolist()
                        objek_terdeteksi = class_names[np.argmax(nilai_prediksi)]

                        # Buat Grafik
                        grafik = go.Figure([go.Bar(x=list(class_names.values()), y=nilai_prediksi)])
                        grafik.update_layout(title='Tingkat Keyakinan Prediksi', xaxis_title='Mineral', yaxis_title='Keyakinan')

                        # Tampilkan Hasil
                        st.write(f"Mineral terdeteksi: {objek_terdeteksi}")
                        st.plotly_chart(grafik)
                    else:
                        st.error("Tidak ada objek yang terdeteksi dalam gambar.")

                except Exception as e:
                    st.error("Gambar tidak dapat terdeteksi.")
                    st.error(f"Error: {e}")

                finally:
                    # Hapus file sementara
                    shutil.rmtree(temp_dir, ignore_errors=True)

st.markdown(
    "<div style='text-align: center;' class='footer'>Program Aplikasi deteksi batuan@2025</div>",
    unsafe_allow_html=True
)
