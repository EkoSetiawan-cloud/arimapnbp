import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def compare_rolling_models_page():
    st.header("🔥 PERBANDINGAN ROLLING FORECAST ARIMA vs ETS")

    # Cek hasil rolling forecast ARIMA & ETS
    df_arima = st.session_state.get('rolling_eval_result_arima')
    df_ets = st.session_state.get('rolling_eval_result_ets')
    if (df_arima is None) or (df_ets is None):
        st.warning("Hasil rolling forecast ARIMA & ETS belum tersedia. Jalankan step 11 & 12 terlebih dahulu.")
        return

    # Merge by Tahun
    df_compare = pd.merge(df_arima, df_ets, on="Tahun", suffixes=('_ARIMA', '_ETS'))
    df_compare = df_compare.rename(columns={
        'Actual_ARIMA': 'Actual'  # asumsikan actual sama untuk kedua model
    })
    st.subheader("Preview Data Gabungan")
    st.write(df_compare)
    
    # Evaluasi metrik
    actual = df_compare['Actual']
    arima_pred = df_compare['Forecast_ARIMA']
    ets_pred = df_compare['Forecast_ETS']
    
    evals = []
    for model, y_pred in zip(['ARIMA', 'ETS'], [arima_pred, ets_pred]):
        mae = mean_absolute_error(actual, y_pred)
        rmse = np.sqrt(mean_squared_error(actual, y_pred))
        mape = mean_absolute_percentage_error(actual, y_pred)
        r2 = r2_score(actual, y_pred)
        evals.append({
            'Model': model,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE (%)': mape,
            'R2': r2
        })
    df_eval = pd.DataFrame(evals)
    st.subheader("Tabel Evaluasi Model")
    st.write(df_eval)
    
    # Plot Perbandingan Aktual, ARIMA, ETS
    st.subheader("Plot Aktual vs Prediksi (ARIMA & ETS)")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df_compare['Tahun'], actual, marker='o', label='Actual', linewidth=3, color='black')
    ax.plot(df_compare['Tahun'], arima_pred, marker='x', label='ARIMA', linestyle='--')
    ax.plot(df_compare['Tahun'], ets_pred, marker='s', label='ETS', linestyle='-.')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Total PNBP')
    ax.set_title('Perbandingan Rolling Forecast')
    ax.legend()
    st.pyplot(fig)

    # Plot Residual ARIMA vs ETS
    st.subheader("Plot Residual (Prediksi - Aktual)")
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(df_compare['Tahun'], arima_pred-actual, marker='x', label='Residual ARIMA')
    ax2.plot(df_compare['Tahun'], ets_pred-actual, marker='s', label='Residual ETS')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_xlabel('Tahun')
    ax2.set_ylabel('Residual')
    ax2.set_title('Residual Model')
    ax2.legend()
    st.pyplot(fig2)

    # Download ke Excel
    st.subheader("Download Hasil")
    excel = pd.ExcelWriter("perbandingan_rolling_forecast.xlsx", engine='xlsxwriter')
    df_compare.to_excel(excel, sheet_name="Perbandingan", index=False)
    df_eval.to_excel(excel, sheet_name="Evaluasi", index=False)
    excel.close()
    with open("perbandingan_rolling_forecast.xlsx", "rb") as f:
        st.download_button("📥 Download Excel Hasil Perbandingan", data=f, file_name="perbandingan_rolling_forecast.xlsx")
