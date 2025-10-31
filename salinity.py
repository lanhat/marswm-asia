
# Streamlit integration for Salinity LSTM Prediction (Tab 3)
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.markdown("#### LSTM-based Salinity Prediction", unsafe_allow_html=True)
st.write("Upload time-aligned datasets of EC (salinity), discharge (Q), and tide levels.")

# Upload files
uploaded_ec = st.file_uploader("Upload EC Data (.csv)", type=["csv"])
uploaded_q = st.file_uploader("Upload Q (Discharge) Data (.csv)", type=["csv"])
uploaded_tide = st.file_uploader("Upload Tide Data (.csv)", type=["csv"])

if uploaded_ec and uploaded_q and uploaded_tide:
    ec = pd.read_csv(uploaded_ec, header=None).to_numpy().reshape(-1, 1)
    q = pd.read_csv(uploaded_q, header=None).to_numpy().reshape(-1, 1)
    tide = pd.read_csv(uploaded_tide, header=None).to_numpy().reshape(-1, 1)

    # Check alignment
    n = min(len(ec), len(q), len(tide))
    ec, q, tide = ec[:n], q[:n], tide[:n]

    # Normalize
    scaler_ec = MinMaxScaler()
    scaler_q = MinMaxScaler()
    scaler_tide = MinMaxScaler()
    scaled_ec = scaler_ec.fit_transform(ec)
    scaled_q = scaler_q.fit_transform(q)
    scaled_tide = scaler_tide.fit_transform(tide)

    # Create dataset
    past_steps = 12
    future_steps = 6
    blank = 40
    X, y = [], []
    for i in range(past_steps, n - future_steps - blank):
        X.append(np.hstack([
            scaled_ec[i + blank - past_steps:i + blank],
            scaled_q[i - past_steps:i],
            scaled_tide[i + blank - past_steps:i + blank]
        ]))
        y.append(scaled_ec[i + blank:i + blank + future_steps].flatten())
    X = np.array(X).reshape(-1, past_steps, 3)
    y = np.array(y)

    # Load model
    model_path = "output_data/model/251023_1_SW_model.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
        y_pred = model.predict(X)
        y_pred_inv = scaler_ec.inverse_transform(y_pred.reshape(-1, 1))
        y_true_inv = scaler_ec.inverse_transform(y.reshape(-1, 1))

        # Plot
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(y_true_inv, label="Observed", color="blue")
        ax.plot(y_pred_inv, label="Predicted", color="red")
        ax.set_title("Observed vs Predicted EC")
        ax.set_ylabel("EC (mS/cm)")
        ax.set_xlabel("Time Index")
        ax.legend()
        st.pyplot(fig)

        # NSE
        numerator = np.sum((y_true_inv - y_pred_inv)**2)
        denominator = np.sum((y_true_inv - np.mean(y_true_inv))**2)
        nse = 1 - numerator / denominator
        rmse = np.sqrt(np.mean((y_true_inv - y_pred_inv)**2))

        st.success(f"NSE: {nse:.3f}, RMSE: {rmse:.3f}")

    else:
        st.error("Model file not found. Please ensure the model is saved at: output_data/model/251023_1_SW_model.h5")
else:
    st.info("Please upload all required files.")
