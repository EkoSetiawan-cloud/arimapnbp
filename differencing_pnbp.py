import streamlit as st
import pandas as pd

def differencing_page():
    st.header("8️⃣ Differencing Data (Transformasi Stasioner)")
    df = st.session_state.get('train_df')
    if df is not None:
        st.write("Data sebelum differencing:")
        st.dataframe(df)
        diff_series = df['Total_PNBP'].diff().dropna()
        df_diff = pd.DataFrame({
            'Tahun': df['Tahun'].iloc[1:],
            'Total_PNBP_diff': diff_series.values
        })
        st.write("Data setelah differencing (orde 1):")
        st.dataframe(df_diff)
        st.session_state['df_diff'] = df_diff
        st.success("Lanjutkan ke uji stasioneritas pada data hasil differencing.")
    else:
        st.info("Selesaikan split train-test dulu.")
