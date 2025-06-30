import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def grid_search_arima_page():
    st.header("⭐ Grid Search ARIMA (Pilih Model Otomatis)")
    df_train = st.session_state.get('train_df')
    df_test = st.session_state.get('test_df')
    if df_train is not None and df_test is not None:
        st.write("Pilih rentang parameter grid search:")
        p_range = st.slider("Nilai p", 0, 5, (0, 2))
        d_range = st.slider("Nilai d", 0, 2, (0, 1))
        q_range = st.slider("Nilai q", 0, 5, (0, 2))
        max_iter = st.number_input("Batas kombinasi dicoba", value=30, min_value=1)
        
        train_series = df_train['Total_PNBP']
        test_series = df_test['Total_PNBP']
        results = []

        if st.button("Mulai Grid Search!"):
            n_combo = 0
            st.write("Grid search berjalan...")
            prog = st.progress(0.0)
            for p in range(p_range[0], p_range[1]+1):
                for d in range(d_range[0], d_range[1]+1):
                    for q in range(q_range[0], q_range[1]+1):
                        if n_combo >= max_iter:
                            break
                        try:
                            history = list(train_series)
                            predictions = []
                            for t in range(len(test_series)):
                                model = ARIMA(history, order=(p, d, q))
                                model_fit = model.fit()
                                yhat = model_fit.forecast()[0]
                                predictions.append(yhat)
                                history.append(test_series.iloc[t])
                            rmse = np.sqrt(mean_squared_error(test_series, predictions))
                            results.append({"p": p, "d": d, "q": q, "RMSE": rmse})
                        except Exception as e:
                            results.append({"p": p, "d": d, "q": q, "RMSE": np.nan})
                        n_combo += 1
                        prog.progress(n_combo / ( (p_range[1]-p_range[0]+1)*(d_range[1]-d_range[0]+1)*(q_range[1]-q_range[0]+1) ))
            st.success("Grid search selesai!")

            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values("RMSE")
            st.dataframe(df_results)
            if not df_results["RMSE"].isnull().all():
                best = df_results.iloc[0]
                st.info(f"Parameter terbaik: p={best['p']}, d={best['d']}, q={best['q']} (RMSE={best['RMSE']:.2f})")
                st.session_state['arima_best_params'] = (int(best['p']), int(best['d']), int(best['q']))
            else:
                st.warning("Tidak ada parameter valid ditemukan (coba perlebar rentang/grid).")
    else:
        st.info("Selesaikan step split train-test dulu.")
