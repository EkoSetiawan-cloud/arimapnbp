def residual_analysis_page():
    st.header("Residual Analysis (Analisis Error Prediksi)")
    model_name = st.session_state.get('model_last_used', 'Belum ada model')
    st.info(f"Residual yang dianalisis berasal dari model: **{model_name}**")
    df_rolling = st.session_state.get('rolling_eval_result')
    if df_rolling is not None:
        actuals = df_rolling['Actual']
        predictions = df_rolling['Forecast']
        residuals = predictions - actuals
        residuals_clean = residuals.dropna()

        st.write("Nilai residuals:")
        st.write(residuals_clean)

        if len(residuals_clean) == 0:
            st.warning("Semua residual NaN. Model gagal fit di rolling forecast, silakan cek parameter model.")
            return

        st.subheader("Residual (Prediksi - Aktual)")
        st.line_chart(residuals_clean.values)

        st.subheader("Density Plot of Residuals")
        fig2, ax2 = plt.subplots()
        pd.Series(residuals_clean).plot(kind='kde', ax=ax2)
        st.pyplot(fig2)

        st.subheader("Statistik Residual:")
        st.write(pd.Series(residuals_clean).describe())

        st.subheader("Autocorrelation Plot (ACF) of Residuals")
        fig3, ax3 = plt.subplots()
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals_clean, lags=min(8, len(residuals_clean)//2-1), ax=ax3)
        st.pyplot(fig3)
    else:
        st.info("Jalankan ARIMA atau ETS rolling forecast dulu.")
