# Streamlit page: XTAISAL – Salinity LSTM (guided by your baseline script)
# ------------------------------------------------------------
# Author: e-Asia project (Y. Natsuki; M. Kimura; Y. Sato; L. T. Hà; D. Hùng)
# ------------------------------------------------------------
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


st.set_page_config(page_title="XTAISAL – Salinity LSTM", layout="wide")
st.title("XTAISAL – LSTM Dự báo Độ mặn (EC)")
st.caption("e-Asia project (Y. Natsuki; M. Kimura; Y. Sato; L. T. Hà; D. Hùng)")

# --------------------------- helpers ---------------------------
@st.cache_data(show_spinner=False)
def _safe_read_csv(file) -> np.ndarray:
    """Read first column as numeric column, interpolate and return (N,1) float array."""
    s = pd.read_csv(file, header=None).iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    s = s.interpolate(limit_direction="both").ffill().bfill()
    a = s.to_numpy(dtype=float).reshape(-1, 1)
    if a.size == 0:
        raise ValueError("Empty or non-numeric CSV.")
    return a

def _align_minlen(*arrs):
    m = min(a.shape[0] for a in arrs)
    return tuple(a[:m] for a in arrs)

def create_dataset_f2(salinity, Q, Tide, past_steps, future_steps, blank=30):
    X, y = [], []
    n = len(salinity)
    for i in range(past_steps, n - future_steps - blank):
        X.append(np.hstack([Q[i - past_steps:i],
                            Tide[i + blank - past_steps - 1:i + blank - 1]]))
        y.append(salinity[i + blank: i + blank + future_steps].flatten())
    return np.array(X), np.array(y)

def create_dataset_f3(salinity, Q, Tide, past_steps, future_steps, blank=40):
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

def build_lstm(input_steps, n_features, output_steps, lr=1e-3, units=64, dropout=0.2):
    model = tf.keras.models.Sequential([
        LSTM(units, input_shape=(input_steps, n_features), return_sequences=False, dropout=dropout),
        Dense(32, activation='relu'),
        Dense(output_steps)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def _metrics(obs, pred):
    obs = np.asarray(obs).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    L = min(len(obs), len(pred))
    obs, pred = obs[:L], pred[:L]
    mse = float(np.nanmean((obs - pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.nanmean(np.abs(obs - pred)))
    denom = float(np.nansum((obs - np.nanmean(obs)) ** 2))
    nse = np.nan if denom == 0 else float(1.0 - np.nansum((obs - pred) ** 2) / denom)
    return mse, rmse, mae, nse

# --------------------------- sidebar controls ---------------------------
with st.sidebar:
    st.markdown("### Cấu hình mô hình")
    mode = st.selectbox("Chế độ dự báo", ["12h–6h", "12h–12h"])
    past_steps, future_steps = (12, 6) if mode == "12h–6h" else (12, 12)
    blank = st.number_input("Lead (blank) [giờ]", min_value=0, max_value=240, value=40, step=1)

    feature_mode = st.selectbox(
        "Tập biến đầu vào",
        ["f2: [Q, Tide] → EC", "f3: [EC, Q, Tide] → EC",
         "f3_1: [WL, Q, Tide] → EC", "f4: [EC, WL, Q, Tide] → EC"],
        index=3
    )

    st.markdown("---")
    st.markdown("### Huấn luyện / Tải mô hình")
    train_new = st.checkbox("Huấn luyện trong ứng dụng", value=True)
    model_file = st.file_uploader("Hoặc tải mô hình .h5", type=["h5"])
    with st.expander("Siêu tham số huấn luyện"):
        ep = st.number_input("Epochs", 1, 2000, 150, 1)
        bs = st.number_input("Batch size", 1, 2048, 32, 1)
        lr = float(st.number_input("Learning rate", 1e-6, 1e-1, 1e-3, format="%.6f"))
        units = st.number_input("LSTM units", 8, 512, 64, 8)
        dropout = st.slider("Dropout", 0.0, 0.8, 0.2, 0.05)
        patience = st.number_input("EarlyStopping patience", 1, 50, 5)
        min_delta = float(st.number_input("EarlyStopping min_delta", 0.0, 1.0, 1e-4, format="%.6f"))

# --------------------------- data inputs ---------------------------
st.subheader("Dữ liệu đầu vào")
st.write("Tải các tệp CSV **một cột** (EC, Q, Tide; tuỳ chọn WD/WL).")
col1, col2, col3, col4 = st.columns(4)
with col1:
    up_ec = st.file_uploader("EC (salinity)", type=["csv"])
with col2:
    up_q = st.file_uploader("Discharge Q", type=["csv"])
with col3:
    up_tide = st.file_uploader("Tide level", type=["csv"])
with col4:
    up_wd = st.file_uploader("Water depth/level (tuỳ chọn)", type=["csv"])

with st.expander("Chuyển đổi Độ sâu → Mực nước (nếu cần)"):
    st.caption("WL = WD − cao trình đáy (đơn vị nhất quán).")
    rb_ec = st.number_input("Cao trình đáy (mm) tại trạm EC", value=-1236, step=1)
    use_depth_to_wl = st.checkbox("Áp dụng WD → WL cho cột WD/WL đã tải", value=True)

run = st.button("Chạy pipeline XTAISAL")

# --------------------------- pipeline ---------------------------
if run:
    try:
        # 1) Đọc & căn độ dài
        if not (up_ec and up_q and up_tide):
            st.error("Vui lòng tải đủ EC, Q, Tide.")
            st.stop()

        ec = _safe_read_csv(up_ec)
        q = _safe_read_csv(up_q)
        tide = _safe_read_csv(up_tide)

        wl = None
        if up_wd is not None:
            wd = _safe_read_csv(up_wd)
            wl = wd - rb_ec if use_depth_to_wl else wd

        if wl is None:
            ec, q, tide = _align_minlen(ec, q, tide)
        else:
            ec, q, tide, wl = _align_minlen(ec, q, tide, wl)

        N = len(ec)
        if N - (past_steps + future_steps + blank) <= 0:
            st.error("Chuỗi quá ngắn với các tham số hiện tại. Giảm 'blank' hoặc thu nhỏ future/past.")
            st.stop()

        # 2) Chuẩn hoá
        scaler_ec = MinMaxScaler()
        scaler_q = MinMaxScaler()
        scaler_tide = MinMaxScaler()
        scaler_wl = MinMaxScaler() if wl is not None else None

        ec_s = scaler_ec.fit_transform(ec)
        q_s = scaler_q.fit_transform(q)
        tide_s = scaler_tide.fit_transform(tide)
        wl_s = scaler_wl.fit_transform(wl) if wl is not None else None

        # 3) Tạo tập dữ liệu theo lựa chọn
        creator = feature_mode.split(":")[0]
        if creator == "f2":
            X, Y = create_dataset_f2(ec_s, q_s, tide_s, past_steps, future_steps, blank=int(blank))
            n_features = 2
            X = X.reshape(X.shape[0], past_steps, 2)
        elif creator == "f3":
            X, Y = create_dataset_f3(ec_s, q_s, tide_s, past_steps, future_steps, blank=int(blank))
            n_features = 3
            X = X.reshape(X.shape[0], past_steps, 3)
        elif creator == "f3_1":
            if wl_s is None:
                st.error("f3_1 cần WL. Vui lòng tải WD/WL.")
                st.stop()
            X, Y = create_dataset_f3_1(ec_s, wl_s, q_s, tide_s, past_steps, future_steps, blank=int(blank))
            n_features = 3
            X = X.reshape(X.shape[0], past_steps, 3)
        else:
            if wl_s is None:
                st.error("f4 cần WL. Vui lòng tải WD/WL.")
                st.stop()
            X, Y = create_dataset_f4(ec_s, wl_s, q_s, tide_s, past_steps, future_steps, blank=int(blank))
            n_features = 4
            X = X.reshape(X.shape[0], past_steps, 4)

        if X.size == 0:
            st.error("Không đủ mẫu sau khi tạo cửa sổ trượt.")
            st.stop()

        # 4) Chia 6:2:2
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)
        st.write(f"**Shapes:** X_train {X_train.shape} | X_val {X_val.shape} | X_test {X_test.shape}")

        # 5) Huấn luyện hoặc tải mô hình
        if train_new:
            model = build_lstm(past_steps, n_features, future_steps, lr=lr, units=int(units), dropout=float(dropout))
            es = tf.keras.callbacks.EarlyStopping(
                min_delta=float(min_delta), patience=int(patience), mode="auto", restore_best_weights=True
            )
            hist = model.fit(
                X_train, y_train,
                epochs=int(ep), batch_size=int(bs),
                validation_data=(X_val, y_val),
                callbacks=[es], verbose=0
            )

            # Loss plot
            fig_loss, ax = plt.subplots(figsize=(7,3))
            ax.plot(hist.history["loss"], label="Train")
            ax.plot(hist.history["val_loss"], label="Val")
            ax.set_title("Loss (MSE)")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3); ax.legend()
            st.pyplot(fig_loss, use_container_width=True)

            # Save
            os.makedirs("output_data/model", exist_ok=True)
            out_model = f"output_data/model/XTAISAL_{creator}_{mode.replace('–','-')}.h5"
            model.save(out_model, include_optimizer=False)
            st.success(f"Đã lưu mô hình: `{out_model}`")
        else:
            if model_file is None:
                st.error("Chọn huấn luyện hoặc tải mô hình .h5.")
                st.stop()
            model = load_model(model_file)
            # Warn if input shape mismatch
            need_steps = model.input_shape[-2]
            need_feats = model.input_shape[-1]
            if need_steps != past_steps or need_feats != n_features:
                st.warning(f"Mô hình yêu cầu shape ({need_steps}, {need_feats}) nhưng bạn đang dùng ({past_steps}, {n_features}).")

        # 6) Dự báo & nghịch chuẩn
        y_val_pred = model.predict(X_val, verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)

        y_val_true_flat  = scaler_ec.inverse_transform(y_val.reshape(-1,1)).reshape(-1)
        y_val_pred_flat  = scaler_ec.inverse_transform(y_val_pred.reshape(-1,1)).reshape(-1)
        y_test_true_flat = scaler_ec.inverse_transform(y_test.reshape(-1,1)).reshape(-1)
        y_test_pred_flat = scaler_ec.inverse_transform(y_test_pred.reshape(-1,1)).reshape(-1)

        # Làm sạch & cân chiều dài
        def _pair_clean(a, b):
            mask = np.isfinite(a) & np.isfinite(b)
            a, b = a[mask], b[mask]
            L = min(len(a), len(b))
            return a[:L], b[:L]

        y_val_true_flat,  y_val_pred_flat  = _pair_clean(y_val_true_flat,  y_val_pred_flat)
        y_test_true_flat, y_test_pred_flat = _pair_clean(y_test_true_flat, y_test_pred_flat)

        if len(y_val_true_flat) == 0 or len(y_test_true_flat) == 0:
            st.error("Không đủ dữ liệu hữu hạn để tính chỉ số. Kiểm tra lại CSV hoặc tham số cửa sổ.")
            st.stop()

        # 7) Metrics
        mse_v, rmse_v, mae_v, nse_v = _metrics(y_val_true_flat,  y_val_pred_flat)
        mse_t, rmse_t, mae_t, nse_t = _metrics(y_test_true_flat, y_test_pred_flat)

        st.subheader("Chỉ số đánh giá")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Validation**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MSE",  f"{mse_v:.3f}")
            m2.metric("RMSE", f"{rmse_v:.3f}")
            m3.metric("MAE",  f"{mae_v:.3f}")
            m4.metric("NSE",  f"{nse_v:.3f}" if np.isfinite(nse_v) else "NaN")
        with c2:
            st.markdown("**Test**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MSE",  f"{mse_t:.3f}")
            m2.metric("RMSE", f"{rmse_t:.3f}")
            m3.metric("MAE",  f"{mae_t:.3f}")
            m4.metric("NSE",  f"{nse_t:.3f}" if np.isfinite(nse_t) else "NaN")

        # 8) Plot (test)
        fig_cmp, axc = plt.subplots(figsize=(10,3))
        axc.plot(y_test_true_flat,  label="Observed EC", color="#1f77b4")
        axc.plot(y_test_pred_flat,  label="Predicted EC", color="#d62728", alpha=0.9)
        axc.set_title(f"So sánh EC (Test) | {creator} | {mode}")
        axc.set_ylabel("EC (mS/cm)"); axc.set_xlabel("Chỉ số thời gian")
        axc.grid(True, alpha=0.3); axc.legend()
        st.pyplot(fig_cmp, use_container_width=True)

        # 9) Download
        L_all = min(len(y_val_true_flat), len(y_val_pred_flat), len(y_test_true_flat), len(y_test_pred_flat))
        out_df = pd.DataFrame({
            "EC_observed_val":   y_val_true_flat[:L_all],
            "EC_predicted_val":  y_val_pred_flat[:L_all],
            "EC_observed_test":  y_test_true_flat[:L_all],
            "EC_predicted_test": y_test_pred_flat[:L_all],
        })
        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        st.download_button(
            "Tải kết quả (CSV)",
            data=buf.getvalue(),
            file_name=f"XTAISAL_{creator}_{mode.replace('–','-')}_results.csv",
            mime="text/csv",
        )

        with st.expander("Thông tin debug"):
            st.write({
                "N_raw": int(N),
                "past_steps": past_steps, "future_steps": future_steps, "blank": blank,
                "X": X.shape, "Y": Y.shape,
                "X_train": X_train.shape, "X_val": X_val.shape, "X_test": X_test.shape,
                "val_len": len(y_val_true_flat), "test_len": len(y_test_true_flat)
            })

    except Exception as e:
        st.error(f"Pipeline error: {type(e).__name__}: {e}")

st.info("⚙️ Gợi ý môi trường: dùng TensorFlow bản **2.15.x** (Py3.10/3.11) hoặc **2.20.0** (CPU) trên Streamlit Cloud. "
        "Tránh 2.16.1 với Py3.13 (không có wheel). Cần `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tensorflow`.")
