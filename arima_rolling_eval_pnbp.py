import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt

def arima_rolling_eval_page():
    st.header("🔟 Fitting ARIMA, Rolling Forecast & Evaluasi")
    df_train = st.session_state.get('train_df')
    df_test = st.session_state.get('test_df')
    df_diff = st.session_state.get('df_diff')
    if df_train is not None and df_test is not None and df_diff is not None:
        p = st.number_input("Nilai p (AR):", min_value=0, value=1)
        d = 1  # Orde differencing
        q = st.number_input("Nilai q (MA):", min_value=0, value=1)
        st.info("Model ARIMA({}, {}, {}) akan di-fit pada data train".format(p,d,q))

        train_series = df_train['Total_PNBP']
        test_series = df_test['Total_PNBP']

        # Rolling forecast untuk seluruh tahun test
        history = list(train_series)
        predictions = []
        log_pred = []
        for t in range(len(test_series)):
            model = ARIMA(history, order=(p,d,q))
            model_fit = model.fit()
            output = model_fit.forecast()
            pred = output[0]
            predictions.append(pred)
            obs = test_series.iloc[t]
            history.append(obs)
            log_pred.append((df_test['Tahun'].iloc[t], pred, obs))
            st.write(f"Step {t+1}: Tahun {df_test['Tahun'].iloc[t]}, predicted={pred:.2f}, expected={obs:.2f}")

        # Evaluasi
        actuals = test_series.values
        mae = np.mean(np.abs(np.array(predictions) - actuals))
        rmse = np.sqrt(np.mean((np.array(predictions) - actuals) ** 2))
        mape = np.mean(np.abs((np.array(predictions) - actuals) / actuals)) * 100

        st.write("## Evaluasi Hasil Prediksi")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAPE: {mape:.2f}%")
        
        df_eval = pd.DataFrame({
            "Tahun": df_test['Tahun'],
            "Actual": actuals,
            "Forecast": predictions
        })
        st.write(df_eval)
        st.session_state['rolling_eval_result'] = df_eval
        st.session_state['model_last_used'] = 'ARIMA'

        # Plot
        fig, ax = plt.subplots()
        ax.plot(df_eval["Tahun"], df_eval["Actual"], label="Actual", marker="o")
        ax.plot(df_eval["Tahun"], df_eval["Forecast"], label="Forecast", marker="o")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Total PNBP")
        ax.set_title("Forecast vs Actual (Rolling Forecast)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Selesaikan step sebelumnya (train-test & differencing) dulu.")
