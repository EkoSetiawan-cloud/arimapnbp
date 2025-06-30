import streamlit as st
from statsmodels.tsa.stattools import adfuller

def stationarity_diff_page():
    st.header("9️⃣ Uji Stasioneritas (ADF Test) Setelah Differencing")
    df_diff = st.session_state.get('df_diff')
    if df_diff is not None:
        series = df_diff['Total_PNBP_diff']
        st.write("Menjalankan ADF Test pada data setelah differencing...")
        result = adfuller(series)
        st.write(f"ADF Statistic: {result[0]:.4f}")
        st.write(f"p-value: {result[1]:.4f}")
        for key, value in result[4].items():
            st.write(f'Critical Value ({key}): {value:.4f}')
        if result[1] < 0.05:
            st.success("Data sudah stasioner! (p-value < 0.05)")
        else:
            st.warning("Data **masih belum** stasioner (p-value >= 0.05). Bisa coba differencing orde 2 jika perlu.")
        st.session_state['adf_pvalue_diff'] = result[1]
    else:
        st.info("Selesaikan step differencing dulu.")
