# salinity_streamlit.py
# ------------------------------------------------------------
# XTAISAL – LSTM-based Salinity Prediction (Streamlit version)
# e-Asia project (Y. Natsuki; M. Kimura; Y. Sato; L. T. Hà; D. Hùng)
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


# --------------------------- Page setup ---------------------------
st.set_page_config(page_title="XTAISAL – Salinity LSTM", layout="wide")
st.title("XTAISAL – Mô hình LSTM dự báo độ mặn (EC)")
st.caption("e-Asia project (Y. Natsuki; M. Kimura; Y. Sato; L. T. Hà; D. Hùng)")


# --------------------------- Utilities (adapted from your script) ---------------------------
def get_range(df: pd.DataFrame, period_mini: str, period_max: str) -> pd.DataFrame:
    """
    Cắt theo chuỗi ký tự thời gian (ở cột 0) giống logic gốc.
    Yêu cầu file có cột 0 là chuỗi thời gian đúng định dạng với period_mini / period_max.
    """
    try:
        i0 = df.index[df[0] == period_mini][0]
    except IndexError:
        raise ValueError(f"'{period_mini}' không tìm thấy trong cột 0.")
    try:
        i1 = df.index[df[0] == period_max][0]
    except IndexError:
        raise ValueError(f"'{period_max}' không tìm thấy trong cột 0.")
    return df.iloc[i0:i1, :]


def create_dataset_f2(salinity, Q, Tide, past_steps, future_steps, blank=30):
    X, y = [], []
    n = len(salinity)
    for i in range(past_steps, n - future_steps - blank):
        X.append(np.hstack([
            Q[i - past_steps:i],
            Tide[i + blank - past_steps - 1:i + blank - 1]
        ]))
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


def cal_nse(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    a = np.sum((y_true - y_pred) ** 2)
    b = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if b == 0 else 1 - (a / b)


def pair_clean(obs, pred):
    obs = np.asarray(obs).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    L = min(len(obs), len(pred))
    obs = obs[:L]
    pred = pred[:L]
    mask = np.isfinite(obs) & np.isfinite(pred)
    return obs[mask], pred[mask]


def plot2_series(y_true, y_pred, title="Observed vs Predicted Salinity Over Time"):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(y_true, label="Observed", color="#1f77b4")
    ax.plot(y_pred, label="Predicted", color="#d62728", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("EC (mS/cm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


# --------------------------- Sidebar: configuration ---------------------------
with st.sidebar:
    st.markdown("### Cấu hình")
    mode = st.selectbox("Chế độ dự báo", ["12h–6h", "12h–12h"])
    past_steps, future_steps = (12, 6) if mode == "12h–6h" else (12, 12)
    blank = st.number_input("Lead (blank) [giờ]", min_value=0, max_value=240, value=40, step=1)

    feature_mode = st.selectbox(
        "Tập biến đầu vào",
        [
            "f2: [Q, Tide] → EC",
            "f3: [EC, Q, Tide] → EC",
            "f3_1: [WL, Q, Tide] → EC",
            "f4: [EC, WL, Q, Tide] → EC"
        ],
        index=3
    )

    st.divider()
    st.markdown("### Huấn luyện / Tải mô hình")
    train_new = st.checkbox("Huấn luyện trong ứng dụng", value=True)
    uploaded_model = st.file_uploader("Hoặc tải mô hình (.h5)", type=["h5"])

    with st.expander("Siêu tham số"):
        ep = st.number_input("Epochs", 1, 2000, 150, 1)
        bs = st.number_input("Batch size", 1, 2048, 32, 1)
        lr = float(st.number_input("Learning rate", 1e-6, 1e-1, 1e-3, format="%.6f"))
        units = st.number_input("LSTM units", 8, 512, 64, 8)
        dropout = st.slider("Dropout", 0.0, 0.8, 0.2, 0.05)
        patience = st.number_input("EarlyStopping patience", 1, 50, 5)
        min_delta = float(st.number_input("EarlyStopping min_delta", 0.0, 1.0, 1e-4, format="%.6f"))


# --------------------------- Inputs ---------------------------
st.subheader("Dữ liệu đầu vào")
st.write("Tải các tệp CSV **một cột**. Đối với **ts_catxuyen.csv** và **ts_connam.csv**, file cần cột 0 là chuỗi thời gian (để dùng `get_range`).")

col = st.columns(3)
with col[0]:
    up_ts_catxuyen = st.file_uploader("`ts_catxuyen.csv` (cột 0: thời gian; cột 1: WD; cột 3: EC)", type=["csv"])
with col[1]:
    up_q_viettri = st.file_uploader("`Q_Viettri_2409-2505.csv` (cột 1: Q)", type=["csv"])
with col[2]:
    up_ts_connam = st.file_uploader("`ts_connam.csv` (cột 0: thời gian; cột 1: Tide)", type=["csv"])

st.markdown("**Khoảng thời gian cắt (matching chuỗi thời gian trong cột 0):**")
period_mini_bf = st.text_input("period_mini_bf (30-min data start)", "10/01/2024 12:00:00 AM")
period_max_bf  = st.text_input("period_max_bf (30-min data end)",   "02/22/2025 11:30:00 PM")
period_mini_af = st.text_input("period_mini_af (10-min data start)", "02/23/2025 12:00:00 AM")
period_max_af  = st.text_input("period_max_af (10-min data end)",   "04/30/2025 11:10:00 PM")

period_mini_Q  = st.text_input("period_mini_Q", "10/01/2024 12:00:00 AM")
period_max_Q   = st.text_input("period_max_Q",  "04/30/2025 11:00:00 PM")

st.markdown("**Cao trình đáy (mm) để chuyển WD→WL:**")
colb = st.columns(2)
with colb[0]:
    riverbed_elevation_8 = st.number_input("Cao trình đáy tại trạm EC (mm)", value=-1236, step=1)
with colb[1]:
    riverbed_elevation_Tide = st.number_input("Cao trình đáy tại trạm triều (mm)", value=-1251, step=1)

run = st.button("Chạy pipeline")


# --------------------------- Pipeline ---------------------------
if run:
    try:
        # Load CSVs (raw)
        if not (up_ts_catxuyen and up_q_viettri and up_ts_connam):
            st.error("Vui lòng tải đủ 3 file: ts_catxuyen.csv, Q_Viettri_2409-2505.csv, ts_connam.csv.")
            st.stop()

        df_8 = pd.read_csv(up_ts_catxuyen, header=None, skiprows=4)
        df_Q = pd.read_csv(up_q_viettri, header=None)
        df_T = pd.read_csv(up_ts_connam, header=None, skiprows=4)

        # Giống script: chọn cột dùng
        # df_8: [date_str, WD, EC] tại cột [0,1,3]; df_T: [date_str, Tide] tại cột [0,1]
        # (nếu không đủ cột, báo lỗi)
        if df_8.shape[1] < 4:
            raise ValueError("`ts_catxuyen.csv` cần >= 4 cột (0:time,1:WD,3:EC).")
        if df_T.shape[1] < 2:
            raise ValueError("`ts_connam.csv` cần >= 2 cột (0:time,1:Tide).")
        df_8 = df_8.iloc[:, [0, 1, 3]]
        df_T = df_T.iloc[:, [0, 1]]

        # 30-min → 1h (lấy cách 2 dòng) cho giai đoạn trước (bf)
        df_8_bf = get_range(df_8, period_mini_bf, period_max_bf).reset_index(drop=True)
        df_T_bf = get_range(df_T, period_mini_bf, period_max_bf).reset_index(drop=True)
        df_8_ec_bf = df_8_bf.iloc[::2, 2].reset_index(drop=True)  # EC
        df_8_wd_bf = df_8_bf.iloc[::2, 1].reset_index(drop=True)  # WD
        df_tide_bf = df_T_bf.iloc[::2, 1].reset_index(drop=True)  # Tide

        # 10-min → 1h (lấy cách 6 dòng) cho giai đoạn sau (af)
        df_8_af = get_range(df_8, period_mini_af, period_max_af).reset_index(drop=True)
        df_T_af = get_range(df_T, period_mini_af, period_max_af).reset_index(drop=True)
        df_8_ec_af = df_8_af.iloc[::6, 2].reset_index(drop=True)
        df_8_wd_af = df_8_af.iloc[::6, 1].reset_index(drop=True)
        df_tide_af = df_T_af.iloc[::6, 1].reset_index(drop=True)

        # Gộp hai giai đoạn
        df_8_ec = pd.concat([df_8_ec_bf, df_8_ec_af]).reset_index(drop=True)
        df_8_wd = pd.concat([df_8_wd_bf, df_8_wd_af]).reset_index(drop=True)
        df_tide = pd.concat([df_tide_bf, df_tide_af]).reset_index(drop=True)

        # Q theo range
        df_Q_rng = get_range(df_Q, period_mini_Q, period_max_Q).reset_index(drop=True)
        if df_Q_rng.shape[1] < 2:
            raise ValueError("`Q_Viettri_2409-2505.csv` cần >= 2 cột (cột 1 là Q).")
        df_Q_1col = df_Q_rng.iloc[:, 1].reset_index(drop=True)

        # Align min length
        m = min(len(df_8_ec), len(df_8_wd), len(df_tide), len(df_Q_1col))
        df_8_ec = df_8_ec.iloc[:m]
        df_8_wd = df_8_wd.iloc[:m]
        df_tide = df_tide.iloc[:m]
        df_Q_1col = df_Q_1col.iloc[:m]

        # Numpy column vectors
        ec = df_8_ec.to_numpy(dtype=float).reshape(-1, 1)
        wd = df_8_wd.to_numpy(dtype=float).reshape(-1, 1)
        tide = df_tide.to_numpy(dtype=float).reshape(-1, 1)
        q = df_Q_1col.to_numpy(dtype=float).reshape(-1, 1)

        # Độ sâu → mực nước (WL) theo cao trình đáy
        wl_ec = wd - riverbed_elevation_8  # mm-based, giống script
        wl_tide = tide - riverbed_elevation_Tide  # nếu cần dùng ở nơi khác

        # Chuẩn hoá
        scaler_ec = MinMaxScaler()
        scaler_wl = MinMaxScaler()
        scaler_Q = MinMaxScaler()
        scaler_T = MinMaxScaler()

        scaled_ec = scaler_ec.fit_transform(ec)
        scaled_wl = scaler_wl.fit_transform(wl_ec)
        scaled_q = scaler_Q.fit_transform(q)
        scaled_tide = scaler_T.fit_transform(wl_tide)  # giữ cách làm tương tự “reshaped_data_T”

        # Chọn dataset builder
        creator = feature_mode.split(":")[0]
        if creator == "f2":
            X, Y = create_dataset_f2(scaled_ec, scaled_q, scaled_tide, past_steps, future_steps, blank)
            n_features = 2
            X = X.reshape(X.shape[0], past_steps, 2)
        elif creator == "f3":
            X, Y = create_dataset_f3(scaled_ec, scaled_q, scaled_tide, past_steps, future_steps, blank)
            n_features = 3
            X = X.reshape(X.shape[0], past_steps, 3)
        elif creator == "f3_1":
            X, Y = create_dataset_f3_1(scaled_ec, scaled_wl, scaled_q, scaled_tide, past_steps, future_steps, blank)
            n_features = 3
            X = X.reshape(X.shape[0], past_steps, 3)
        else:
            X, Y = create_dataset_f4(scaled_ec, scaled_wl, scaled_q, scaled_tide, past_steps, future_steps, blank)
            n_features = 4
            X = X.reshape(X.shape[0], past_steps, 4)

        if X.size == 0:
            st.error("Không đủ mẫu sau khi tạo cửa sổ (past_steps/future_steps/blank). Hãy điều chỉnh lại tham số.")
            st.stop()

        # Chia 6:2:2 (train:val:test)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)

        st.success(f"Shapes → X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

        # Huấn luyện hoặc tải model
        if train_new:
            model = tf.keras.models.Sequential([
                LSTM(int(units), input_shape=(X_train.shape[1], X_train.shape[2]),
                     return_sequences=False, dropout=float(dropout)),
                Dense(32, activation='relu'),
                Dense(future_steps)
            ])
            model.compile(optimizer=Adam(learning_rate=float(lr)), loss='mse')

            es = tf.keras.callbacks.EarlyStopping(
                min_delta=float(min_delta),
                patience=int(patience),
                mode='auto',
                restore_best_weights=True
            )

            hist = model.fit(
                X_train, y_train,
                epochs=int(ep), batch_size=int(bs),
                validation_data=(X_val, y_val),
                callbacks=[es],
                verbose=0
            )

            # Plot loss
            fig_loss, ax = plt.subplots(figsize=(7, 3))
            ax.plot(hist.history['loss'], label='Training Loss')
            ax.plot(hist.history['val_loss'], label='Validation Loss')
            ax.set_xlabel('Epochs'); ax.set_ylabel('MSE loss')
            ax.set_title('Training vs Validation Loss'); ax.grid(True, alpha=0.3); ax.legend()
            st.pyplot(fig_loss, use_container_width=True)

            # Save model
            os.makedirs('output_data/model', exist_ok=True)
            out_model = f"output_data/model/XTAISAL_{creator}_{mode.replace('–','-')}_SW_model.h5"
            model.save(out_model, include_optimizer=False)
            st.success(f"Đã lưu mô hình: `{out_model}`")
        else:
            if uploaded_model is None:
                st.error("Bạn chọn không huấn luyện. Vui lòng tải mô hình .h5.")
                st.stop()
            model = load_model(uploaded_model)
            need_steps = model.input_shape[-2]
            need_feats = model.input_shape[-1]
            if (need_steps, need_feats) != (past_steps, n_features):
                st.warning(f"Mô hình cần shape ({need_steps}, {need_feats}) "
                           f"nhưng bạn cấu hình ({past_steps}, {n_features}).")

        # Đánh giá (val) và dự báo (test)
        val_loss = float(model.evaluate(X_val, y_val, verbose=0))
        st.info(f"Validation loss (MSE): {val_loss:.5f}")

        y_val_pred = model.predict(X_val, verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)

        # Nghịch chuẩn về EC
        y_val_true = scaler_ec.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
        y_val_pred_ec = scaler_ec.inverse_transform(y_val_pred.reshape(-1, 1)).reshape(-1)
        y_test_true = scaler_ec.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
        y_test_pred_ec = scaler_ec.inverse_transform(y_test_pred.reshape(-1, 1)).reshape(-1)

        # Clean pairs & metrics
        y_val_true, y_val_pred_ec = pair_clean(y_val_true, y_val_pred_ec)
        y_test_true, y_test_pred_ec = pair_clean(y_test_true, y_test_pred_ec)

        mse_v = float(np.mean((y_val_true - y_val_pred_ec) ** 2))
        rmse_v = float(np.sqrt(mse_v))
        mae_v = float(np.mean(np.abs(y_val_true - y_val_pred_ec)))
        nse_v = cal_nse(y_val_true, y_val_pred_ec)

        mse_t = float(np.mean((y_test_true - y_test_pred_ec) ** 2))
        rmse_t = float(np.sqrt(mse_t))
        mae_t = float(np.mean(np.abs(y_test_true - y_test_pred_ec)))
        nse_t = cal_nse(y_test_true, y_test_pred_ec)

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

        # Plot test
        fig_cmp = plot2_series(y_test_true, y_test_pred_ec,
                               title=f"So sánh EC (Test) | {feature_mode.split(':')[0]} | {mode}")
        st.pyplot(fig_cmp, use_container_width=True)

        # Xuất bảng kết quả
        L = min(len(y_val_true), len(y_val_pred_ec), len(y_test_true), len(y_test_pred_ec))
        out_df = pd.DataFrame({
            "EC_observed_val":  y_val_true[:L],
            "EC_predicted_val": y_val_pred_ec[:L],
            "EC_observed_test": y_test_true[:L],
            "EC_predicted_test": y_test_pred_ec[:L],
        })
        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        st.download_button("Tải kết quả (CSV)",
                           data=buf.getvalue(),
                           file_name=f"XTAISAL_{feature_mode.split(':')[0]}_{mode.replace('–','-')}_results.csv",
                           mime="text/csv")

        with st.expander("Debug info"):
            st.write({
                "X": X.shape, "Y": Y.shape,
                "X_train": X_train.shape, "X_val": X_val.shape, "X_test": X_test.shape
            })

    except Exception as e:
        st.error(f"Lỗi pipeline: {type(e).__name__}: {e}")


# --------------------------- Environment tip ---------------------------
st.info("⚙️ Gợi ý môi trường: dùng TensorFlow **2.15.x** (Py3.10/3.11) hoặc **2.20.0 (CPU)** trên Streamlit Cloud. "
        "Tránh 2.16.1 với Py3.13. Cần `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tensorflow`.")
