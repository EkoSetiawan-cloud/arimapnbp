import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def residual_analysis_page():
    st.header("Residual Analysis (Analisis Error Prediksi)")
    model_name = st.session_state.get('model_last_used', 'Belum ada model')
    st.info(f"Residual yang dianalisis berasal dari model: **{model_name}**")
    df_rolling = st.session_state.get('rolling_eval_result')
    if df_rolling is not None:
        actuals = df_rolling['Actual']
        predictions = df_rolling['Forecast']
        residuals = predictions - actuals

        st.subheader("Residual (Prediksi - Aktual)")
        st.line_chart(residuals.values)

        st.subheader("Density Plot of Residuals")
        fig2, ax2 = plt.subplots()
        pd.Series(residuals).plot(kind='kde', ax=ax2)
        st.pyplot(fig2)

        st.subheader("Statistik Residual:")
        st.write(pd.Series(residuals).describe())

        st.subheader("Autocorrelation Plot (ACF) of Residuals")
        fig3, ax3 = plt.subplots()
        plot_acf(residuals, lags=min(8, len(residuals)//2-1), ax=ax3)
        st.pyplot(fig3)
    else:
        st.info("Jalankan ARIMA atau ETS rolling forecast dulu.")
