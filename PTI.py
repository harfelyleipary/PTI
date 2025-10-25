import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

#nama aplikasi di icon tab
st.set_page_config(page_title="Predictivetive Time Intelligence")
st.title("Aplikasi Evaluasi RealTime : Temperatur dan Kelembapan ")
st.markdown("Prediksi Temperatur & Kelembapan menggunakan model Random Forest, XGBoost, dan LSTM.")

# === Upload CSV baru ===
uploaded_file = st.file_uploader("Upload CSV data sensor", type=["csv"])
if uploaded_file is not None:
    data_new = pd.read_csv(uploaded_file)
    st.write(f"Data berhasil dibaca: {data_new.shape[0]} baris")
    st.dataframe(data_new.head())

    if "Temperature" not in data_new.columns or "Humidity" not in data_new.columns:
        st.error("CSV harus memiliki kolom 'Temperature' dan 'Humidity'")
        st.stop()

    # === Urutkan data berdasarkan waktu jika ada ===
    if 'created_at' in data_new.columns:
        data_new['created_at'] = pd.to_datetime(data_new['created_at'])
        data_new = data_new.sort_values('created_at').reset_index(drop=True)

    # === Load semua model & scaler ===
    save_dir = r"D:/KULIAH/S2/Semester 3/Predictive Time Inteligence - 3/Codingan/Codingan_Tugas_KakYosia/saved_models"
    try:
        # --- Random Forest ---
        rf_temp = joblib.load(os.path.join(save_dir, "model_rf_temperature.pkl"))
        rf_hum  = joblib.load(os.path.join(save_dir, "model_rf_humidity.pkl"))
        if not hasattr(rf_temp, "predict") or not hasattr(rf_hum, "predict"):
            raise TypeError("File RF bukan model, periksa kembali file .pkl")

        # --- XGBoost ---
        xgb_temp = joblib.load(os.path.join(save_dir, "model_xgb_temperature.pkl"))
        xgb_hum  = joblib.load(os.path.join(save_dir, "model_xgb_humidity.pkl"))
        if not hasattr(xgb_temp, "predict") or not hasattr(xgb_hum, "predict"):
            raise TypeError("File XGB bukan model, periksa kembali file .pkl")

        # --- LSTM ---
        lstm_temp = tf.keras.models.load_model(
            os.path.join(save_dir, "model_lstm_temperature.h5"),
            custom_objects={"mse": tf.keras.metrics.MeanSquaredError()}
        )
        lstm_hum = tf.keras.models.load_model(
            os.path.join(save_dir, "model_lstm_humidity.h5"),
            custom_objects={"mse": tf.keras.metrics.MeanSquaredError()}
        )

        # --- Scalers ---
        scaler_X = joblib.load(os.path.join(save_dir, "scaler_X.pkl"))
        scaler_y_temp = joblib.load(os.path.join(save_dir, "scaler_y_temperature.pkl"))
        scaler_y_hum  = joblib.load(os.path.join(save_dir, "scaler_y_humidity.pkl"))

        st.success("Model dan scaler berhasil dimuat.")
    except Exception as e:
        st.error(f"Gagal memuat model/scaler: {e}")
        st.stop()

    # === Buat fitur lag sesuai training ===
    for lag in range(1,4):
        data_new[f'Temp_t-{lag}'] = data_new['Temperature'].shift(lag)
        data_new[f'Hum_t-{lag}']  = data_new['Humidity'].shift(lag)
    data_new = data_new.dropna().reset_index(drop=True)

    # === Fitur untuk prediksi ===
    feature_cols = [f'Temp_t-{i}' for i in range(1,4)] + [f'Hum_t-{i}' for i in range(1,4)]
    X_new = data_new[feature_cols].values
    X_new_scaled = scaler_X.transform(X_new)
    X_new_lstm = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))

    # === Prediksi ===
    preds_rf_temp = rf_temp.predict(X_new_scaled)
    preds_rf_hum  = rf_hum.predict(X_new_scaled)
    preds_xgb_temp = xgb_temp.predict(X_new_scaled)
    preds_xgb_hum  = xgb_hum.predict(X_new_scaled)
    preds_lstm_temp = lstm_temp.predict(X_new_lstm)
    preds_lstm_hum  = lstm_hum.predict(X_new_lstm)

    # === Inverse transform ===
    preds_rf_temp_inv  = scaler_y_temp.inverse_transform(preds_rf_temp.reshape(-1,1))
    preds_rf_hum_inv   = scaler_y_hum.inverse_transform(preds_rf_hum.reshape(-1,1))
    preds_xgb_temp_inv = scaler_y_temp.inverse_transform(preds_xgb_temp.reshape(-1,1))
    preds_xgb_hum_inv  = scaler_y_hum.inverse_transform(preds_xgb_hum.reshape(-1,1))
    preds_lstm_temp_inv = scaler_y_temp.inverse_transform(preds_lstm_temp)
    preds_lstm_hum_inv  = scaler_y_hum.inverse_transform(preds_lstm_hum)

    # === Evaluasi RMSE & R² ===
    def evaluate(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)

    y_temp_true = data_new['Temperature'].values
    y_hum_true  = data_new['Humidity'].values

    st.subheader("Evaluasi Temperature")
    models_temp = {
        "Random Forest": preds_rf_temp_inv,
        "XGBoost": preds_xgb_temp_inv,
        "LSTM": preds_lstm_temp_inv
    }
    for name, pred in models_temp.items():
        rmse, r2 = evaluate(y_temp_true, pred)
        st.write(f"**{name}** → RMSE: `{rmse:.3f}`, R²: `{r2:.3f}`")

    st.subheader("Evaluasi Humidity")
    models_hum = {
        "Random Forest": preds_rf_hum_inv,
        "XGBoost": preds_xgb_hum_inv,
        "LSTM": preds_lstm_hum_inv
    }
    for name, pred in models_hum.items():
        rmse, r2 = evaluate(y_hum_true, pred)
        st.write(f"**{name}** → RMSE: `{rmse:.3f}`, R²: `{r2:.3f}`")

    # === Plot Prediksi ===
    fig, ax = plt.subplots(1,2, figsize=(14,5))

    ax[0].plot(y_temp_true, label="Actual Temp", color='black', linewidth=2)
    ax[0].plot(preds_rf_temp_inv, label="RF", linestyle='--')
    ax[0].plot(preds_xgb_temp_inv, label="XGB", linestyle='-.')
    ax[0].plot(preds_lstm_temp_inv, label="LSTM", linestyle=':')
    ax[0].set_title("Temperature Prediction Comparison")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(y_hum_true, label="Actual Humidity", color='black', linewidth=2)
    ax[1].plot(preds_rf_hum_inv, label="RF", linestyle='--')
    ax[1].plot(preds_xgb_hum_inv, label="XGB", linestyle='-.')
    ax[1].plot(preds_lstm_hum_inv, label="LSTM", linestyle=':')
    ax[1].set_title("Humidity Prediction Comparison")
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)

    # === Simpan Hasil Prediksi ke CSV ===
    data_new["Pred_RF_Temp"] = preds_rf_temp_inv
    data_new["Pred_XGB_Temp"] = preds_xgb_temp_inv
    data_new["Pred_LSTM_Temp"] = preds_lstm_temp_inv
    data_new["Pred_RF_Hum"]  = preds_rf_hum_inv
    data_new["Pred_XGB_Hum"] = preds_xgb_hum_inv
    data_new["Pred_LSTM_Hum"] = preds_lstm_hum_inv

    csv = data_new.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Unduh Hasil Prediksi (CSV)",
        data=csv,
        file_name="hasil_prediksi.csv",
        mime="text/csv"
    )

else:
    st.info("Silakan upload CSV data sensor terlebih dahulu.")
