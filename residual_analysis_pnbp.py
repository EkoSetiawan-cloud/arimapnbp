import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def residual_analysis_page():
    st.header("Residual Analysis (Analisis Error Prediksi)")

    opsi = []
    if 'rolling_eval_result_arima' in st.session_state:
        opsi.append("ARIMA")
    if 'rolling_eval_result_ets' in st.session_state:
        opsi.append("ETS")
    if not opsi:
        st.warning("Belum ada rolling forecast yang dijalankan.")
        return

    model_name = st.selectbox("Pilih model untuk residual analysis:", opsi)

    if model_name == "ARIMA":
        df_rolling = st.session_state.get('rolling_eval_result_arima')
    else:
        df_rolling = st.session_state.get('rolling_eval_result_ets')

    st.write("DEBUG df_rolling")
    st.write(df_rolling)

    if df_rolling is not None and not df_rolling.empty:
        actuals = df_rolling['Actual']
        predictions = df_rolling['Forecast']
        residuals = predictions - actuals
        st.write("DEBUG residuals:")
        st.write(residuals)

        if residuals.isnull().all():
            st.warning("Semua residual NaN. Coba ulangi rolling forecast, atau cek output prediksi.")
            return

        st.subheader("Residual (Prediksi - Aktual)")
        st.line_chart(residuals.values)

        st.subheader("Density Plot of Residuals")
        fig2, ax2 = plt.subplots()
        pd.Series(residuals.dropna()).plot(kind='kde', ax=ax2)
        st.pyplot(fig2)

        st.subheader("Statistik Residual:")
        st.write(pd.Series(residuals.dropna()).describe())

        st.subheader("Autocorrelation Plot (ACF) of Residuals")
        from statsmodels.graphics.tsaplots import plot_acf
        fig3, ax3 = plt.subplots()
        plot_acf(residuals.dropna(), lags=min(8, max(1, len(residuals.dropna())//2-1)), ax=ax3)
        st.pyplot(fig3)
    else:
        st.warning("Data rolling forecast tidak ditemukan atau kosong.")

