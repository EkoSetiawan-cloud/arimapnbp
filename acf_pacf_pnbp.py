import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def acf_pacf_page():
    st.header("7️⃣ Plot ACF & PACF")
    df = st.session_state.get('train_df')
    if df is not None:
        series = df['Total_PNBP'].dropna()
        lags = min(8, len(series)//2 - 1)
        if lags < 1:
            st.warning("Data terlalu sedikit untuk plot ACF/PACF.")
            return
        st.write(f"Plot ACF & PACF dengan lag maksimal {lags}")
        fig1, ax1 = plt.subplots()
        plot_acf(series, lags=lags, ax=ax1)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        plot_pacf(series, lags=lags, ax=ax2)
        st.pyplot(fig2)
    else:
        st.info("Selesaikan split train-test dulu.")
