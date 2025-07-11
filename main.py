import streamlit as st
from input_data_pnbp import input_data_page
from aggregate_data_pnbp import aggregate_data_page
from clean_data_pnbp import clean_data_page
from split_data_pnbp import split_data_page
from eda_pnbp import eda_page
from stationarity_pnbp import stationarity_page
from acf_pacf_pnbp import acf_pacf_page
from differencing_pnbp import differencing_page
from stationarity_diff_pnbp import stationarity_diff_page
from grid_search_arima_pnbp import grid_search_arima_page
from arima_rolling_eval_pnbp import arima_rolling_eval_page
from ets_rolling_eval_pnbp import ets_rolling_eval_page
from residual_analysis_pnbp import residual_analysis_page
from compare_rolling_models_pnbp import compare_rolling_models_page
from kmeans_cluster_pnbp import kmeans_cluster_page





st.set_page_config(page_title="PNBP ARIMA Project", layout="wide")
st.sidebar.title("Prediksi PNBP ARIMA/ETS Rolling Forecast")

steps = [
    "1. Input Data",
    "2. Agregasi Data",
    "3. Bersih Data",
    "4. Split Train-Test",
    "5. Eksplorasi Data (EDA)",
    "6. Uji Stasioneritas",
    "7. Plot ACF & PACF",
    "8. Differencing Data",
    "9. Uji Stasioneritas Setelah Differencing",
    "10. Grid Search ARIMA",
    "11. ARIMA, Rolling Forecast",
    "12. Exponential Smoothing (ETS) Rolling Forecast",
    "13. Residual Analysis",
    "14. Perbandingan ARIMA vs ETS",
    "15. Klastering KMeans"
]
step = st.sidebar.radio("Pilih langkah:", steps)

if step == "1. Input Data":
    input_data_page()
elif step == "2. Agregasi Data":
    aggregate_data_page()
elif step == "3. Bersih Data":
    clean_data_page()
elif step == "4. Split Train-Test":
    split_data_page()
elif step == "5. Eksplorasi Data (EDA)":
    eda_page()
elif step == "6. Uji Stasioneritas":
    stationarity_page()
elif step == "7. Plot ACF & PACF":
    acf_pacf_page()
elif step == "8. Differencing Data":
    differencing_page()
elif step == "9. Uji Stasioneritas Setelah Differencing":
    stationarity_diff_page()
elif step == "10. Grid Search ARIMA":
    grid_search_arima_page()
elif step == "11. ARIMA, Rolling Forecast":
    arima_rolling_eval_page()
elif step == "12. Exponential Smoothing (ETS) Rolling Forecast":
    ets_rolling_eval_page()
elif step == "13. Residual Analysis":
    residual_analysis_page()
elif step == "14. Perbandingan ARIMA vs ETS":
    compare_rolling_models_page()
elif step == "15. Klastering KMeans":
    kmeans_cluster_page()


st.sidebar.markdown("---")
st.sidebar.info("Jalankan modul berurutan untuk hasil optimal.")
