import streamlit as st
from statsmodels.tsa.stattools import adfuller

def stationarity_page():
    st.header("6️⃣ Uji Stasioneritas (ADF Test)")
    df = st.session_state.get('train_df')
    if df is not None:
        series = df['Total_PNBP']
        st.write("Menjalankan ADF Test...")
        result = adfuller(series)
        st.write(f"ADF Statistic: {result[0]:.4f}")
        st.write(f"p-value: {result[1]:.4f}")
        for key, value in result[4].items():
            st.write(f'Critical Value ({key}): {value:.4f}')
        if result[1] < 0.05:
            st.success("Data stasioner! (p-value < 0.05)")
        else:
            st.warning("Data **belum** stasioner (p-value >= 0.05), lakukan differencing di modul berikutnya.")
        st.session_state['adf_pvalue'] = result[1]
    else:
        st.info("Selesaikan split train-test dulu.")
