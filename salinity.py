# ======================= XTAISAL – LSTM Salinity Module =======================
# Streamlit UI + full pipeline (preprocess → train/load → evaluate → visualize)
# Authors: e-Asia project (Y. Natsuki; M. Kimura; Y. Sato; L. T. Hà; D. Hùng)

import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Soft deps (installed in your env):
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# TensorFlow import with friendly message if not available
try:
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
except Exception as e:
    st.error(
        "TensorFlow is not available in this environment. "
        "Install a compatible version (e.g., tensorflow==2.15.0 or 2.20.0 for Python ≥3.11)."
    )
    st.stop()

st.markdown("### Salinity Model (LSTM)")
st.caption("XTAISAL – e-Asia project (Y. Natsuki; M. Kimura; Y. Sato; L. T. Hà; D. Hùng)")

# ------------------------------ Helpers ---------------------------------------
def safe_read_csv(file) -> pd.Series:
    """Read CSV first column → numeric Series, interpolate NaNs both ways."""
    s = pd.read_csv(file, header=None).iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    s = s.interpolate(limit_direction="both").ffill().bfill()
    return s

def to_col(x, name="var"):
    arr = pd.Series(x).astype(float).to_numpy().reshape(-1, 1)
    if arr.size == 0:
        raise ValueError(f"[{name}] empty array after reading.")
    return arr

def align_len(*arrs):
    m = min(a.shape[0] for a in arrs)
    return tuple(a[:m] for a in arrs)

def create_dataset_f2(salinity, Q, Tide, past_steps, future_steps, blank=30):
    # Inputs: [Q, Tide]  Target: EC
    X, y = [], []
    n = len(salinity)
    for i in range(past_steps, n - future_steps - blank):
        X.append(np.hstack([Q[i - past_steps:i],
                            Tide[i + blank - past_steps - 1:i + blank - 1]]))
        y.append(salinity[i + blank: i + blank + future_steps].flatten())
    return np.array(X), np.array(y)

def create_dataset_f3(salinity, Q, Tide, past_steps, future_steps, blank=40):
    # Inputs: [EC, Q, Tide]  Target: EC
    X, y = [], []
    n = len(salinity)
    for i in range(past_steps, n - future_steps - blank):
        X.append(np.hstack([
            salinity[i + blank - past_steps:i + blank],
            Q[i - past_steps:i],
            Tide[i + blank - past_steps:i + blank]
        ]))
        y.append(salinity[i + blank: i + blank + future_steps].flatten())
    return np.array(X), np.array(y)

def create_dataset_f3_1(salinity, water_level, Q, Tide, past_steps, future_steps, blank=40):
    # Inputs: [WL, Q, Tide]  Target: EC
    X, y = [], []
    n = len(water_level)
    for i in range(past_steps, n - future_steps - blank):
        X.append(np.hstack([
            water_level[i + blank - past_steps:i + blank],
            Q[i - past_steps:i],
            Tide[i + blank - past_steps:i + blank]
        ]))
        y.append(salinity[i + blank: i + blank + future_steps].flatten())
    return np.array(X), np.array(y)

def create_dataset_f4(salinity, water_level, Q, Tide, past_steps, future_steps, blank=40):
    # Inputs: [EC, WL, Q, Tide]  Target: EC
    X, y = [], []
    n = len(water_level)
    for i in range(past_steps, n - future_steps - blank):
        X.append(np.hstack([
            salinity[i + blank - past_steps:i + blank],
            water_level[i + blank - past_steps:i + blank],
            Q[i - past_steps:i],
            Tide[i + blank - past_steps:i + blank]
        ]))
        y.append(salinity[i + blank: i + blank + future_steps].flatten())
    return np.array(X), np.array(y)

def metrics(obs, pred):
    obs = obs.reshape(-1)
    pred = pred.reshape(-1)
    mse  = np.mean((obs - pred) ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(obs - pred))
    denom = np.sum((obs - np.mean(obs)) ** 2)
    nse  = np.nan if denom == 0 else 1 - np.sum((obs - pred) ** 2) / denom
    return mse, rmse, mae, nse

def build_lstm(input_steps, n_features, output_steps, lr=1e-3, units=64, dropout=0.2):
    model = tf.keras.models.Sequential([
        LSTM(units, input_shape=(input_steps, n_features),
             return_sequences=False, dropout=dropout),
        Dense(32, activation='relu'),
        Dense(output_steps)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

# --------------------------- Controls / Options --------------------------------
colA, colB, colC = st.columns([1,1,1])
with colA:
    mode = st.selectbox("Forecast mode", ["12h–6h", "12h–12h"])
    past_steps, future_steps = (12, 6) if mode == "12h–6h" else (12, 12)

with colB:
    blank = st.number_input("Lead (blank) [hours]", min_value=0, max_value=240, value=40, step=1)

with colC:
    feature_mode = st.selectbox(
        "Feature set",
        [
            "f2: [Q, Tide] → EC",
            "f3: [EC, Q, Tide] → EC",
            "f3_1: [WL, Q, Tide] → EC",
            "f4: [EC, WL, Q, Tide] → EC"
        ],
        index=3
    )

st.divider()

st.markdown("**Upload time-aligned CSV (1 column each)**")
up_ec   = st.file_uploader("EC (salinity)", type=["csv"])
up_q    = st.file_uploader("Discharge Q", type=["csv"])
up_tide = st.file_uploader("Tide level", type=["csv"])
up_wd   = st.file_uploader("(Optional) Water depth (for WL conversion)", type=["csv"])

with st.expander("Depth → Water Level (WL) conversion"):
    st.caption("Nếu có số liệu độ sâu (WD), chuyển sang mực nước (WL) theo cao trình đáy.")
    riverbed_ec   = st.number_input("Riverbed elevation at EC station (mm)", value=-1236, step=1)
    riverbed_tide = st.number_input("Riverbed elevation at Tide station (mm)", value=-1251, step=1)
    use_depth_to_wl = st.checkbox("Convert WD→WL using riverbed_ec", value=True)

st.markdown("**Model**")
colM1, colM2 = st.columns([1,1])
with colM1:
    train_new = st.checkbox("Train in-app", value=True, help="Huấn luyện LSTM trong ứng dụng.")
with colM2:
    model_file = st.file_uploader("Or upload trained .h5 (skip training)", type=["h5"])

if train_new:
    with st.expander("Training hyperparameters"):
        ep        = st.number_input("Epochs", min_value=1, max_value=2000, value=150, step=1)
        bs        = st.number_input("Batch size", min_value=1, max_value=1024, value=32, step=1)
        lr        = st.number_input("Learning rate", min_value=1e-6, max_value=1e-1,
                                    value=1e-3, step=1e-6, format="%.6f")
        patience  = st.number_input("EarlyStopping patience", min_value=1, max_value=50, value=5, step=1)
        min_delta = st.number_input("EarlyStopping min_delta", min_value=0.0, max_value=1.0,
                                    value=1e-4, step=1e-4, format="%.6f")
        units     = st.number_input("LSTM units", min_value=8, max_value=512, value=64, step=8)
        dropout   = st.slider("Dropout", min_value=0.0, max_value=0.8, value=0.2, step=0.05)
else:
    ep = bs = lr = patience = min_delta = units = dropout = None

# --------------------------------- Run -----------------------------------------
run = st.button("Run Salinity Pipeline")
if run:
    try:
        # 1) Inputs
        if not (up_ec and up_q and up_tide):
            st.error("Vui lòng tải đủ EC, Q, Tide.")
            st.stop()

        ec   = to_col(safe_read_csv(up_ec),   "EC")
        q    = to_col(safe_read_csv(up_q),    "Q")
        tide = to_col(safe_read_csv(up_tide), "Tide")

        wl = None
        if up_wd is not None:
            wd = to_col(safe_read_csv(up_wd), "WD")
            wl = wd - riverbed_ec if use_depth_to_wl else wd

        # 2) Align
        if wl is None:
            ec, q, tide = align_len(ec, q, tide)
        else:
            ec, q, tide, wl = align_len(ec, q, tide, wl)

        # 3) Normalize
        scaler_ec   = MinMaxScaler()
        scaler_q    = MinMaxScaler()
        scaler_tide = MinMaxScaler()
        scaler_wl   = MinMaxScaler() if wl is not None else None

        ec_s   = scaler_ec.fit_transform(ec)
        q_s    = scaler_q.fit_transform(q)
        tide_s = scaler_tide.fit_transform(tide)
        wl_s   = scaler_wl.fit_transform(wl) if wl is not None else None

        # 4) Build dataset
        creator_name = feature_mode.split(":")[0]
        b = int(blank)

        if creator_name == "f2":
            X, Y = create_dataset_f2(ec_s, q_s, tide_s, past_steps, future_steps, blank=b)
            n_features = 2
        elif creator_name == "f3":
            X, Y = create_dataset_f3(ec_s, q_s, tide_s, past_steps, future_steps, blank=b)
            n_features = 3
        elif creator_name == "f3_1":
            if wl_s is None:
                st.error("f3_1 cần Water Level (WL). Vui lòng tải lên WD/WL.")
                st.stop()
            X, Y = create_dataset_f3_1(ec_s, wl_s, q_s, tide_s, past_steps, future_steps, blank=b)
            n_features = 3
        else:  # f4
            if wl_s is None:
                st.error("f4 cần Water Level (WL). Vui lòng tải lên WD/WL.")
                st.stop()
            X, Y = create_dataset_f4(ec_s, wl_s, q_s, tide_s, past_steps, future_steps, blank=b)
            n_features = 4

        # reshape to (samples, past_steps, n_features)
        if creator_name == "f2":
            X = X.reshape(X.shape[0], past_steps, 2)
        elif creator_name in ("f3", "f3_1"):
            X = X.reshape(X.shape[0], past_steps, 3)
        else:
            X = X.reshape(X.shape[0], past_steps, 4)

        if X.size == 0:
            st.error("Không đủ dữ liệu sau khi áp dụng past_steps/future_steps/blank.")
            st.stop()

        # 5) Split 6:2:2
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
        X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)
        st.write(f"**Shapes** → X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

        # 6) Train or load
        if train_new:
            model = build_lstm(past_steps, n_features, future_steps, lr=lr, units=int(units), dropout=float(dropout))
            es = tf.keras.callbacks.EarlyStopping(
                min_delta=float(min_delta), patience=int(patience), mode="auto", restore_best_weights=True
            )
            history = model.fit(
                X_train, y_train,
                epochs=int(ep),
                batch_size=int(bs),
                validation_data=(X_val, y_val),
                callbacks=[es],
                verbose=0
            )

            fig_loss, axl = plt.subplots(figsize=(7,3))
            axl.plot(history.history["loss"], label="Train loss")
            axl.plot(history.history["val_loss"], label="Val loss")
            axl.set_title("Training vs Validation Loss (MSE)")
            axl.set_xlabel("Epoch")
            axl.set_ylabel("Loss")
            axl.grid(True, alpha=0.3)
            axl.legend()
            st.pyplot(fig_loss, use_container_width=True)

            os.makedirs("output_data/model", exist_ok=True)
            out_model_path = f"output_data/model/XTAISAL_{creator_name}_{mode.replace('–','-')}_SW_model.h5"
            model.save(out_model_path, include_optimizer=False)
            st.success(f"Model saved to: `{out_model_path}`")
        else:
            if model_file is None:
                st.error("Chọn *Train in-app* hoặc tải lên mô hình .h5.")
                st.stop()
            model = load_model(model_file)
            if model.input_shape[-2] != past_steps or model.input_shape[-1] != n_features:
                st.warning(
                    f"Mô hình yêu cầu input shape (past_steps={model.input_shape[-2]}, n_features={model.input_shape[-1]}), "
                    f"nhưng đang dùng (past_steps={past_steps}, n_features={n_features})."
                )

        # 7) Predict (val + test)
        y_val_pred  = model.predict(X_val,  verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)

        # Inverse to EC units
        y_val_true   = scaler_ec.inverse_transform(y_val.reshape(-1, 1))
        y_val_pred_i = scaler_ec.inverse_transform(y_val_pred.reshape(-1, 1))
        y_test_true  = scaler_ec.inverse_transform(y_test.reshape(-1, 1))
        y_test_pred_i= scaler_ec.inverse_transform(y_test_pred.reshape(-1, 1))

        # 8) Metrics
        mse_v, rmse_v, mae_v, nse_v = metrics(y_val_true,  y_val_pred_i)
        mse_t, rmse_t, mae_t, nse_t = metrics(y_test_true, y_test_pred_i)

        st.markdown("**Validation metrics**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MSE",  f"{mse_v:.3f}")
        c2.metric("RMSE", f"{rmse_v:.3f}")
        c3.metric("MAE",  f"{mae_v:.3f}")
        c4.metric("NSE",  f"{nse_v:.3f}" if not np.isnan(nse_v) else "NaN")

        st.markdown("**Test metrics**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MSE",  f"{mse_t:.3f}")
        c2.metric("RMSE", f"{rmse_t:.3f}")
        c3.metric("MAE",  f"{mae_t:.3f}")
        c4.metric("NSE",  f"{nse_t:.3f}" if not np.isnan(nse_t) else "NaN")

        # 9) Plot Observed vs Predicted (Test)
        fig_cmp, axc = plt.subplots(figsize=(9,3))
        axc.plot(y_test_true,   label="Observed EC",  color="#1f77b4")
        axc.plot(y_test_pred_i, label="Predicted EC", color="#d62728", alpha=0.9)
        axc.set_title(f"Observed vs Predicted (Test) | {creator_name} | {mode}")
        axc.set_ylabel("EC (mS/cm)")
        axc.set_xlabel("Time Index")
        axc.grid(True, alpha=0.3)
        axc.legend()
        st.pyplot(fig_cmp, use_container_width=True)

        # 10) Download results
        out_df = pd.DataFrame({
            "EC_observed_val":   y_val_true.reshape(-1),
            "EC_predicted_val":  y_val_pred_i.reshape(-1),
            "EC_observed_test":  y_test_true.reshape(-1),
            "EC_predicted_test": y_test_pred_i.reshape(-1),
        })
        buff = io.StringIO()
        out_df.to_csv(buff, index=False)
        st.download_button(
            "Download results (CSV)",
            data=buff.getvalue(),
            file_name=f"XTAISAL_{creator_name}_{mode.replace('–','-')}_results.csv",
            mime="text/csv",
        )

        with st.expander("Debug info"):
            st.write({
                "creator": creator_name,
                "past_steps": past_steps, "future_steps": future_steps, "blank": blank,
                "X.shape": X.shape, "Y.shape": Y.shape,
                "X_train": X_train.shape, "X_val": X_val.shape, "X_test": X_test.shape,
                "test_len_true": len(y_test_true), "test_len_pred": len(y_test_pred_i)
            })

    except Exception as e:
        st.error(f"Pipeline error: {type(e).__name__}: {e}")

# --------------------------- Env notes -----------------------------------------
st.info(
    "📌 Env tips: Pin TensorFlow to a version with wheels for your Python. "
    "For Python ≥3.11, try `tensorflow==2.15.0` (CPU) or `2.20.0`. "
    "Also include `scikit-learn`, `pandas`, `numpy`, `matplotlib` in requirements."
)
