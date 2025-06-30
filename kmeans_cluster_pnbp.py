import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import matplotlib.pyplot as plt

def kmeans_cluster_page():
    st.header("🔬 KMeans Klastering PNBP")

    # Load data dari session_state (dari modul sebelumnya)
    df = st.session_state.get('df_clean')
    if df is None:
        st.warning("Data belum tersedia. Selesaikan step 'Bersih Data' dulu.")
        return

    st.subheader("Preview Data")
    st.dataframe(df)

    # Buat fitur tambahan (misal Growth)
    if 'Growth' not in df.columns:
        df = df.copy()
        df['Growth'] = df['Total_PNBP'].pct_change().fillna(0)

    # Pilih fitur klastering
    fitur_opsi = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != 'Tahun']
    fitur_dipilih = st.multiselect("Pilih fitur untuk klastering:", fitur_opsi, default=['Total_PNBP','Growth'])

    if len(fitur_dipilih) < 2:
        st.info("Pilih minimal 2 fitur untuk visualisasi scatter plot.")
        return

    # Pilih jumlah klaster
    k = st.slider("Jumlah klaster (K):", min_value=2, max_value=6, value=3)

    # Standardisasi data
    scaler = StandardScaler()
    X = scaler.fit_transform(df[fitur_dipilih])

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    df['Cluster'] = cluster_labels

    st.subheader("Hasil Klastering")
    st.dataframe(df[['Tahun'] + fitur_dipilih + ['Cluster']])

    # Visualisasi cluster (2 fitur teratas)
    st.subheader("Visualisasi Klastering (Scatter Plot)")
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df[fitur_dipilih[0]],
        df[fitur_dipilih[1]],
        c=df['Cluster'],
        cmap='viridis',
        s=100
    )
    for i, row in df.iterrows():
        ax.text(row[fitur_dipilih[0]], row[fitur_dipilih[1]], str(row['Tahun']), fontsize=9)
    ax.set_xlabel(fitur_dipilih[0])
    ax.set_ylabel(fitur_dipilih[1])
    ax.set_title('Klastering PNBP dengan KMeans')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    st.pyplot(fig)

    # Download hasil
    st.subheader("Download Hasil Klastering")
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    st.download_button(
        label="📥 Download Excel",
        data=output,
        file_name="hasil_klaster_pnbp.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success(f"Klastering dengan KMeans selesai. Temukan insight dari tiap klaster untuk knowledge-based system kamu!")

