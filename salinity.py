# salinity_lstm_app.py
# Streamlit app: XTAISAL – LSTM-based Salinity Pipeline with Upload/URL inputs
# e-Asia project (Y. Natsuki; M. Kimura; Y. Sato; L. T. Hà; D. Hùng)

import io
import os
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import chardet

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Optional: silence TF logs before importing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="XTAISAL – Salinity (LSTM)", layout="centered")
st.title("XTAISAL – LSTM-based Salinity Prediction")
st.caption("e-Asia project (Y. Natsuki; M. Kimura; Y. Sato; L. T. Hà; D. Hùng)")
st.write(
    "Tải dữ liệu theo **2 cách**: *Upload từ máy* hoặc *nhập URL (raw CSV)*. "
    "Nếu dùng GitHub, **hãy dùng RAW URL**; nếu dùng Google Drive/private link, vui lòng tải file về rồi upload."
)

# ============================ Helpers (robust I/O) ============================

UA = {"User-Agent": "Mozilla/5.0 (XTAISAL/1.0)"}

def _smart_read_df(file_or_text, *, skiprows=0, name="CSV"):
    """
    Đọc CSV với tự động phát hiện encoding & dấu phân tách.
    - Chấp nhận: file-like (Streamlit uploader) hoặc text (từ URL).
    - Trả về: DataFrame (giữ nguyên tất cả cột).
    """
    try:
        # Chuẩn hoá thành bytes buffer để chardet/đọc nhiều lần
        if hasattr(file_or_text, "read"):  # UploadedFile / file-like
            raw = file_or_text.read()
            if not isinstance(raw, (bytes, bytearray)):
                # Trong một số runtime, read() có thể trả về str
                raw = str(raw).encode("utf-8", errors="ignore")
        elif isinstance(file_or_text, str):
            raw = file_or_text.encode("utf-8", errors="ignore")
        else:
            raise ValueError("Đầu vào không phải file-like hoặc text.")

        enc = chardet.detect(raw)["encoding"] or "utf-8"
        buf = io.BytesIO(raw)

        # Thử engine=python sep=None để auto detect; nếu fail, thử ; \t ,
        tried = []
        try:
            df = pd.read_csv(buf, skiprows=skiprows, engine="python", sep=None, encoding=enc)
            tried.append("sep=None")
        except Exception:
            buf.seek(0)
            for sep in [";", "\t", ","]:
                try:
                    df = pd.read_csv(buf, skiprows=skiprows, sep=sep, encoding=enc)
                    tried.append(f"sep='{sep}'")
                    break
                except Exception:
                    buf.seek(0)
            else:
                raise ValueError("Không xác định được dấu phân tách.")
        # Bỏ cột hoàn toàn rỗng
        df = df.dropna(how="all", axis=1)
        if df.shape[1] == 0:
            raise ValueError("File không có cột dữ liệu sau khi loại cột rỗng.")
        return df
    except Exception as e:
        raise ValueError(f"Lỗi đọc {name}: {type(e).__name__}: {e}")

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Làm sạch & chuyển chuỗi → số:
    - bỏ khoảng trắng, ký tự không phải số (giữ - và .)
    - nếu chỉ có dấu phẩy (,) thì coi là dấu thập phân → đổi thành .
    - nội suy & lấp khoảng trống 2 đầu
    """
    s = s.astype(str).str.strip()

    has_comma = s.str.contains(",", regex=False).any()
    has_dot   = s.str.contains(r"\.", regex=True).any()
    if has_comma and not has_dot:
        # ví dụ: 1,23  →  1.23
        s = s.str.replace(",", ".", regex=False)

    # bỏ ký tự không phải số/thập phân/dấu âm
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    s = pd.to_numeric(s, errors="coerce")
    s = s.interpolate(limit_direction="both").ffill().bfill()
    return s

def _fetch_url_text(url: str) -> str:
    """GET text với user-agent & thông báo lỗi thân thiện (403/401...)."""
    try:
        r = requests.get(url, headers=UA, timeout=25)
        if r.status_code == 200:
            return r.text
        hint = ""
        if "github.com" in url and "raw.githubusercontent.com" not in url:
            hint = " (Bạn đang dùng URL trang HTML. Hãy dùng **raw.githubusercontent.com/...**)."
        if "drive.google.com" in url:
            hint = " (Link Drive có thể riêng tư/không direct. Hãy tải về rồi upload.)"
        raise ValueError(f"HTTP {r.status_code}{hint}")
    except requests.RequestException as e:
        raise ValueError(f"Không truy cập được URL: {e}")

def ui_csv_input(label: str, *, skiprows=0, required=True, name="CSV",
                 allow_url=True, default_col_idx=None, col_name_hints=()):
    """
    Widget nhập CSV (Upload/URL) → hiển thị preview + CHỌN CỘT.
    - Tự động phát hiện encoding/sep.
    - Làm sạch số & ép kiểu; nếu tất cả NaN → báo lỗi.
    - default_col_idx: gợi ý cột mặc định (theo chỉ số).
    - col_name_hints: gợi ý theo tên cột ('EC', 'salinity', ...)
    Trả về: ndarray (N,1)
    """
    st.markdown(f"**{label}**")
    opts = ["Upload từ máy"] + (["Nhập URL (raw CSV)"] if allow_url else [])
    how = st.radio(f"Cách nhập {label}", opts, horizontal=True, key=f"how_{label}")

    df = None
    if how == "Upload từ máy":
        up = st.file_uploader(f"Chọn file {label}", type=["csv"], key=f"up_{label}")
        if up is not None:
            df = _smart_read_df(up, skiprows=skiprows, name=name)
    else:
        url = st.text_input(f"URL {label} (raw CSV)", key=f"url_{label}",
                            placeholder="https://raw.githubusercontent.com/...")
        if url:
            try:
                txt = _fetch_url_text(url)
                df = _smart_read_df(txt, skiprows=skiprows, name=f"{name} (URL)")
            except Exception as e:
                st.error(str(e))

    if df is None:
        if required:
            st.info(f"Hãy cung cấp {label}.")
        return None

    st.caption(f"Xem trước 5 dòng đầu của {name}:")
    st.dataframe(df.head(), use_container_width=True)

    # Gợi ý cột theo tên hoặc theo index mặc định
    columns = list(df.columns)
    preselect = None
    if col_name_hints:
        cand = [c for c in columns if str(c).strip().lower() in [h.lower() for h in col_name_hints]]
        if cand:
            preselect = cand[0]
    if preselect is None and default_col_idx is not None and default_col_idx < len(columns):
        preselect = columns[default_col_idx]
    if preselect is None:
        preselect = columns[0]

    col = st.selectbox(f"Chọn cột chứa {name}", options=columns,
                       index=columns.index(preselect), key=f"{name}_col_select")
    series_raw = df[col]
    series_num = _coerce_numeric_series(series_raw)

    if series_num.isna().all():
        st.error(
            f"Không thể trích số cho {name} từ cột `{col}`. "
            "Hãy chọn cột khác hoặc kiểm tra định dạng (dấu `,`/`.`/đơn vị)."
        )
        st.stop()

    arr = series_num.to_numpy().reshape(-1, 1)
    return arr

def to_col(x, name="var"):
    """Series/array → float ndarray (N,1) với kiểm tra cơ bản."""
    arr = pd.Series(x.ravel() if isinstance(x, np.ndarray) else x).astype(float).to_numpy().reshape(-1, 1)
    if arr.size == 0:
        raise ValueError(f"[{name}] mảng rỗng sau khi đọc.")
    return arr

def align_len(*arrs):
    """Cắt mọi mảng về cùng độ dài nhỏ nhất."""
    m = min(a.shape[0] for a in arrs)
    return tuple(a[:m] for a in arrs)

# ====================== Dataset creators (your functions) ======================

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

def metrics(obs, pred):
    obs = obs.reshape(-1)
    pred = pred.reshape(-1)
    if len(obs) == 0 or len(pred) == 0:
        return np.nan, np.nan, np.nan, np.nan
    if len(obs) != len(pred):
        m = min(len(obs), len(pred))
        obs, pred = obs[:m], pred[:m]
    mse = np.mean((obs - pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(obs - pred))
    denom = np.sum((obs - np.mean(obs)) ** 2)
    nse = np.nan if denom == 0 else 1 - np.sum((obs - pred) ** 2) / denom
    return mse, rmse, mae, nse

def build_lstm(input_steps, n_features, output_steps, lr=1e-3, units=64, dropout=0.2):
    model = tf.keras.models.Sequential([
        LSTM(units, input_shape=(input_steps, n_features), return_sequences=False, dropout=dropout),
        Dense(32, activation='relu'),
        Dense(output_steps)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

# ============================ UI: Inputs & Options ============================

st.subheader("1) Nhập dữ liệu")
# Theo cấu trúc các file gốc đã nêu:
# - ts_catxuyen.csv: skiprows=4, EC thường ở cột index 2 (0-based)
# - Q_Viettri_2409-2505.csv: Q ở cột index 1
# - ts_connam.csv: skiprows=4, Tide ở cột index 1

s_ec = ui_csv_input(
    "EC (salinity) – ts_catxuyen.csv",
    skiprows=2,                     # was 4 → now 3 header lines
    name="EC",
    allow_url=True,
    default_col_idx=3,              # EC is the 4th column after timestamp
    col_name_hints=("ec", "salinity", "conductivity", "ec_ms_cm", "ec_mscm"),
)

s_q = ui_csv_input(
    "Discharge Q – Q_Viettri_2409-2505.csv",
    skiprows=0, name="Q",
    allow_url=True,
    default_col_idx=1,
    col_name_hints=("q", "discharge", "flow", "q_m3s"),
)

s_tide = ui_csv_input(
    "Tide level – ts_connam.csv",
    skiprows=2,                     # was 4 → align to 3 header lines like EC file
    name="Tide",
    allow_url=True,
    default_col_idx=1,
    col_name_hints=("tide", "wl", "water_level", "tide_level"),
)

st.markdown("**(Tuỳ chọn)**: Nạp **độ sâu** để chuyển sang **mực nước (WL)**")
s_wd = ui_csv_input("Water Depth (WD) – optional",
                    skiprows=4, name="WD",
                    allow_url=True, required=False,
                    default_col_idx=1,
                    col_name_hints=("wd", "depth", "water_depth"))

with st.expander("Thiết lập chuyển đổi độ sâu → mực nước (WD→WL)"):
    riverbed_ec = st.number_input("Cao trình đáy (EC station, mm)", value=-1236, step=1)
    riverbed_tide = st.number_input("Cao trình đáy (Tide station, mm)", value=-1251, step=1)
    use_depth_to_wl = st.checkbox("Chuyển WD → WL (WL = WD - cao trình đáy EC)", value=True)

st.subheader("2) Tuỳ chọn dự báo & đặc trưng")
mode = st.selectbox("Chế độ dự báo", ["12h–6h", "12h–12h"])
past_steps, future_steps = (12, 6) if mode == "12h–6h" else (12, 12)
blank = st.number_input("Lead (blank) [giờ]", min_value=0, max_value=240, value=40, step=1)

feature_mode = st.selectbox(
    "Tập đặc trưng (dataset builder)",
    [
        "f2: [Q, Tide] → EC",
        "f3: [EC, Q, Tide] → EC",
        "f3_1: [WL, Q, Tide] → EC",
        "f4: [EC, WL, Q, Tide] → EC"
    ],
    index=3
)

st.subheader("3) Mô hình")
colM1, colM2 = st.columns(2)
with colM1:
    train_new = st.checkbox("Huấn luyện trong ứng dụng", value=True)
with colM2:
    model_file = st.file_uploader("Hoặc tải mô hình .h5 đã huấn luyện", type=["h5"])

if train_new:
    with st.expander("Siêu tham số huấn luyện"):
        ep = st.number_input("Epochs", 1, 2000, 150, 1)
        bs = st.number_input("Batch size", 1, 1024, 32, 1)
        lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-1, value=1e-3, step=1e-6, format="%.6f")
        patience = st.number_input("EarlyStopping patience", 1, 50, 5, 1)
        min_delta = st.number_input("EarlyStopping min_delta", min_value=0.0, max_value=1.0, value=1e-4, step=1e-4, format="%.6f")
        units = st.number_input("LSTM units", 8, 512, 64, 8)
        dropout = st.slider("Dropout", 0.0, 0.8, 0.2, 0.05)
else:
    ep = bs = lr = patience = min_delta = units = dropout = None

# ============================== Run the pipeline ==============================

if st.button("🚀 Chạy pipeline dự báo mặn (LSTM)"):
    try:
        if s_ec is None or s_q is None or s_tide is None:
            st.error("Vui lòng cung cấp **đủ** EC, Q và Tide.")
            st.stop()

        # 1) To columns
        ec   = to_col(s_ec,   "EC")
        q    = to_col(s_q,    "Q")
        tide = to_col(s_tide, "Tide")

        wl = None
        if s_wd is not None:
            wd = to_col(s_wd, "WD")
            wl = wd - riverbed_ec if use_depth_to_wl else wd

        # 2) Align lengths
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
        if creator_name == "f2":
            X, Y = create_dataset_f2(ec_s, q_s, tide_s, past_steps, future_steps, blank=int(blank))
            n_features = 2
        elif creator_name == "f3":
            X, Y = create_dataset_f3(ec_s, q_s, tide_s, past_steps, future_steps, blank=int(blank))
            n_features = 3
        elif creator_name == "f3_1":
            if wl_s is None:
                st.error("f3_1 yêu cầu Water Level (WL). Vui lòng cung cấp WD/WL.")
                st.stop()
            X, Y = create_dataset_f3_1(ec_s, wl_s, q_s, tide_s, past_steps, future_steps, blank=int(blank))
            n_features = 3
        else:  # f4
            if wl_s is None:
                st.error("f4 yêu cầu Water Level (WL). Vui lòng cung cấp WD/WL.")
                st.stop()
            X, Y = create_dataset_f4(ec_s, wl_s, q_s, tide_s, past_steps, future_steps, blank=int(blank))
            n_features = 4

        if X.size == 0:
            st.error("Không đủ dữ liệu sau khi áp dụng past_steps/future_steps/blank.")
            st.stop()

        X = X.reshape(X.shape[0], past_steps, n_features)
        Y = Y.reshape(-1, future_steps)

        # 5) Split 6:2:2
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
        X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)

        st.write(f"**Shapes** → X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

        # 6) Train or load model
        if train_new:
            model = build_lstm(past_steps, n_features, future_steps, lr=float(lr),
                               units=int(units), dropout=float(dropout))
            es = tf.keras.callbacks.EarlyStopping(
                min_delta=float(min_delta), patience=int(patience), mode="auto", restore_best_weights=True
            )
            history = model.fit(
                X_train, y_train,
                epochs=int(ep), batch_size=int(bs),
                validation_data=(X_val, y_val),
                callbacks=[es],
                verbose=0
            )

            # plot loss
            fig_loss, ax = plt.subplots(figsize=(7,3))
            ax.plot(history.history["loss"], label="Train loss")
            ax.plot(history.history["val_loss"], label="Val loss")
            ax.set_title("Training vs Validation Loss (MSE)")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3); ax.legend()
            st.pyplot(fig_loss, use_container_width=True)

            # save model
            os.makedirs("output_data/model", exist_ok=True)
            out_model_path = f"output_data/model/XTAISAL_{creator_name}_{mode.replace('–','-')}_SW_model.h5"
            model.save(out_model_path, include_optimizer=False)
            st.success(f"Đã lưu mô hình: `{out_model_path}`")

        else:
            if model_file is None:
                st.error("Chọn *Huấn luyện trong ứng dụng* hoặc tải mô hình .h5.")
                st.stop()
            model = load_model(model_file)
            if model.input_shape[-2] != past_steps or model.input_shape[-1] != n_features:
                st.warning(
                    f"Mô hình .h5 yêu cầu input (steps={model.input_shape[-2]}, features={model.input_shape[-1]}) "
                    f"không khớp (steps={past_steps}, features={n_features})."
                )

        # 7) Predict & inverse scale
        y_val_pred  = model.predict(X_val,  verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)

        y_val_true   = scaler_ec.inverse_transform(y_val.reshape(-1, 1))
        y_val_pred_i = scaler_ec.inverse_transform(y_val_pred.reshape(-1, 1))
        y_test_true  = scaler_ec.inverse_transform(y_test.reshape(-1, 1))
        y_test_pred_i= scaler_ec.inverse_transform(y_test_pred.reshape(-1, 1))

        # 8) Metrics
        mse_v, rmse_v, mae_v, nse_v = metrics(y_val_true,  y_val_pred_i)
        mse_t, rmse_t, mae_t, nse_t = metrics(y_test_true, y_test_pred_i)

        st.markdown("**Validation metrics**")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("MSE",  f"{mse_v:.3f}" if not np.isnan(mse_v) else "NaN")
        c2.metric("RMSE", f"{rmse_v:.3f}" if not np.isnan(rmse_v) else "NaN")
        c3.metric("MAE",  f"{mae_v:.3f}" if not np.isnan(mae_v) else "NaN")
        c4.metric("NSE",  f"{nse_v:.3f}" if not np.isnan(nse_v) else "NaN")

        st.markdown("**Test metrics**")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("MSE",  f"{mse_t:.3f}" if not np.isnan(mse_t) else "NaN")
        c2.metric("RMSE", f"{rmse_t:.3f}" if not np.isnan(rmse_t) else "NaN")
        c3.metric("MAE",  f"{mae_t:.3f}" if not np.isnan(mae_t) else "NaN")
        c4.metric("NSE",  f"{nse_t:.3f}" if not np.isnan(nse_t) else "NaN")

        # 9) Plot
        fig_cmp, axc = plt.subplots(figsize=(9,3))
        axc.plot(y_test_true,  label="EC quan trắc", color="#1f77b4")
        axc.plot(y_test_pred_i,label="EC dự báo",   color="#d62728", alpha=0.9)
        axc.set_title(f"So sánh EC (Test) | {creator_name} | {mode}")
        axc.set_ylabel("EC (mS/cm)"); axc.set_xlabel("Chỉ số thời gian")
        axc.grid(True, alpha=0.3); axc.legend()
        st.pyplot(fig_cmp, use_container_width=True)

        # 10) Export results
        out_df = pd.DataFrame({
            "EC_observed_val":  y_val_true.reshape(-1),
            "EC_predicted_val": y_val_pred_i.reshape(-1),
            "EC_observed_test": y_test_true.reshape(-1),
            "EC_predicted_test":y_test_pred_i.reshape(-1),
        })
        csv_buf = io.StringIO()
        out_df.to_csv(csv_buf, index=False)
        st.download_button("Tải kết quả (CSV)",
                           data=csv_buf.getvalue(),
                           file_name=f"XTAISAL_{creator_name}_{mode.replace('–','-')}_results.csv",
                           mime="text/csv")

        with st.expander("Thông tin gỡ lỗi"):
            st.write({
                "creator": creator_name,
                "past_steps": past_steps, "future_steps": future_steps, "blank": blank,
                "X.shape": X.shape, "Y.shape": Y.shape,
                "X_train": X_train.shape, "X_val": X_val.shape, "X_test": X_test.shape,
                "test_len_true": len(y_test_true), "test_len_pred": len(y_test_pred_i)
            })

    except Exception as e:
        st.error(f"Lỗi pipeline: {type(e).__name__}: {e}")

# =============================== Environment tip ==============================
st.info(
    "📌 **Trên Streamlit Cloud, ghim phiên bản TensorFlow tương thích Python, "
    "ví dụ `tensorflow==2.15.0` hoặc `2.20.0`. Tránh `2.16.1` với Python 3.13 (không có wheels). "
    "Các thư viện cần: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `requests`, `chardet`."
)
