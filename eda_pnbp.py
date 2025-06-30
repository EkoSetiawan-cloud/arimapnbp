import streamlit as st
import matplotlib.pyplot as plt

def eda_page():
    st.header("5️⃣ Eksplorasi Data (EDA) Total PNBP")
    df = st.session_state.get('df_clean')
    if df is not None:
        st.write("Statistik deskriptif:")
        st.dataframe(df.describe())

        # Plot Time Series
        fig, ax = plt.subplots()
        ax.plot(df['Tahun'], df['Total_PNBP'], marker='o')
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Total PNBP')
        ax.set_title('Trend Total PNBP per Tahun')
        st.pyplot(fig)

        # Plot Histogram
        fig2, ax2 = plt.subplots()
        ax2.hist(df['Total_PNBP'], bins=10)
        ax2.set_title("Distribusi Total PNBP")
        ax2.set_xlabel("Total PNBP")
        st.pyplot(fig2)
    else:
        st.info("Selesaikan split data dulu.")
