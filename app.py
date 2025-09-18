# app.py
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import streamlit as st
from sklearn.neighbors import NearestNeighbors

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

# ---------- Caching helpers ----------
@st.cache_data(show_spinner=False)
def load_weekly_csvs():
    """Load saved weekly trend CSVs if available."""
    data = {}
    for name, fn in {
        "product_group": "weekly_product_group.csv",
        "colour_group": "weekly_colour_group.csv",
        "graphical_appearance": "weekly_graphical_appearance.csv",
    }.items():
        path = ART / fn
        data[name] = pd.read_csv(path, parse_dates=["week"]) if path.exists() else None
    return data

@st.cache_data(show_spinner=False)
def load_meta_features():
    """Load meta and the memmap features (lazy)."""
    meta = pd.read_csv(ART / "meta.csv")
    N, D = len(meta), 1280
    X = np.memmap(ART / "X_img_mm.dat", dtype="float32", mode="r", shape=(N, D))
    return meta, X

@st.cache_data(show_spinner=False)
def load_classifier():
    """Load classifier + class list if saved."""
    clf_path = ART / "logreg_img.pkl"
    classes_path = ART / "classes.json"
    if not (clf_path.exists() and classes_path.exists()):
        return None, None
    clf = joblib.load(clf_path)
    classes = json.load(open(classes_path))["classes"]
    return clf, classes

@st.cache_resource(show_spinner=False)
def load_backbone():
    """Load MobileNetV2 backbone + preprocessing."""
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    # Gentle GPU settings (safe on laptops)
    try:
        for g in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

    tf.keras.backend.clear_session()
    tf.keras.backend.set_image_data_format("channels_last")
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
    )
    IMG_SIZE = (224, 224)

    def preprocess(path_or_image):
        """Takes a Path or PIL Image or bytes and returns a preprocessed tf tensor."""
        import tensorflow as tf
        if isinstance(path_or_image, (str, os.PathLike)):
            raw = tf.io.read_file(str(path_or_image))
            x = tf.image.decode_jpeg(raw, channels=3)
        else:
            # assume PIL.Image.Image
            arr = np.array(path_or_image.convert("RGB"))
            x = tf.convert_to_tensor(arr, dtype=tf.uint8)
        x = tf.image.resize(x, IMG_SIZE)
        x = tf.cast(x, tf.float32)     # 0..255
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        x = preprocess_input(x)        # [-1,1]
        return x

    return base, preprocess

@st.cache_resource(show_spinner=False)
def build_nn_index(X):
    """Build cosine-similar NN index over normalized features."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms
    nn = NearestNeighbors(metric="cosine", n_neighbors=6).fit(Xn)
    return nn

# ---------- UI ----------
st.set_page_config(page_title="Fashion Trend Analyser", layout="wide")
st.title("🧵 Fashion Trend Analyser")

# Load artifacts
if not ART.exists():
    st.error("`artifacts/` folder not found. Run your feature extraction & weekly trend scripts first.")
    st.stop()

weekly = load_weekly_csvs()
meta, X_img = load_meta_features()
clf, class_names = load_classifier()

# Tabs
tab_trend, tab_sim, tab_pred = st.tabs(["📈 Trends", "🖼️ Similarity Search", "🔮 Predict"])

# ----------------- TRENDS -----------------
with tab_trend:
    import pandas as pd

    st.subheader("Weekly Trends")

    kind = st.selectbox(
        "Choose trend type",
        ["product_group", "colour_group", "graphical_appearance"],
        format_func=lambda k: {
            "product_group": "Product Group",
            "colour_group": "Colour Group",
            "graphical_appearance": "Graphical Appearance",
        }[k],
    )

    df = weekly.get(kind)
    if df is None or df.empty:
        st.warning(f"No saved CSV for `{kind}` found in artifacts/, or it is empty. "
                   "Run the chunked aggregation to generate it.")
        st.stop()

    # Ensure proper dtypes
    df = df.copy()
    df["week"] = pd.to_datetime(df["week"], errors="coerce")

    # Map the category column name explicitly instead of relying on column order
    cat_col_map = {
        "product_group": "product_group_name",
        "colour_group": "colour_group_name",
        "graphical_appearance": "graphical_appearance_name",
    }
    cat_col = cat_col_map[kind]
    if cat_col not in df.columns:
        st.error(f"Expected column `{cat_col}` not found in the CSV. "
                 f"Found columns: {list(df.columns)}")
        st.stop()

    # Top N picker
    topN = st.slider("Top N categories to show", 3, 10, 5)

    # Compute top categories overall
    top_cats = (
        df.groupby(cat_col)["sales_count"]
          .sum()
          .sort_values(ascending=False)
          .head(topN)
          .index.tolist()
    )
    dff = df[df[cat_col].isin(top_cats)].copy()
    if dff.empty:
        st.warning("No rows match the selected categories.")
        st.stop()

    # Date slider must use Python date/datetime, not pandas Timestamp
    min_w = pd.to_datetime(dff["week"].min()).to_pydatetime().date()
    max_w = pd.to_datetime(dff["week"].max()).to_pydatetime().date()

    start_date, end_date = st.slider(
        "Date range",
        value=(min_w, max_w),
        min_value=min_w,
        max_value=max_w,
    )

    mask = (dff["week"].dt.date >= start_date) & (dff["week"].dt.date <= end_date)
    dff = dff[mask]
    if dff.empty:
        st.info("No data in the selected date range.")
        st.stop()

    # Pivot to weeks × categories
    pivot = (
        dff.pivot(index="week", columns=cat_col, values="sales_count")
           .sort_index()
           .fillna(0)
    )

    # Show chart + a little table so you can “see the analysis”
    st.line_chart(pivot)  # 👈 the chart
    st.caption("Last 10 rows of the pivot:")
    st.dataframe(pivot.tail(10))

    # Optional extra: quick totals for the selected range
    totals = dff.groupby(cat_col)["sales_count"].sum().sort_values(ascending=False)
    st.write("Totals in selected range:")
    st.dataframe(totals.to_frame("sales_count"))



# ----------------- SIMILARITY -----------------
with tab_sim:
    st.subheader("Find Visually Similar Items")
    base, preprocess = load_backbone()
    nn = build_nn_index(X_img)

    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    topk = st.slider("Neighbors (k)", 3, 12, 6)

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Query", width=256)

        x = preprocess(img)
        import tensorflow as tf
        feat = base(tf.expand_dims(x, 0), training=False).numpy()
        feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
        dists, idxs = nn.kneighbors(feat, n_neighbors=topk)
        res = meta.iloc[idxs[0]][["image_path", "product_group_name"]].assign(distance=dists[0])

        st.write("**Nearest matches:**")
        cols = st.columns(min(topk, 6))
        for j, (_, row) in enumerate(res.iterrows()):
            with cols[j % len(cols)]:
                p = Path(row["image_path"])
                if p.exists():
                    st.image(str(p), use_column_width=True)
                st.caption(f"{row['product_group_name']}  \nDist: {row['distance']:.3f}")
        st.dataframe(res)

# ----------------- PREDICT -----------------
with tab_pred:
    st.subheader("Predict Product Group (using your trained classifier)")
    if clf is None or class_names is None:
        st.info("Classifier/classes not found. Save them first (logreg_img.pkl, classes.json in artifacts/).")
    else:
        base, preprocess = load_backbone()
        f = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
        if f:
            img = Image.open(f)
            st.image(img, caption="Input", width=256)

            x = preprocess(img)
            import tensorflow as tf
            feat = base(tf.expand_dims(x, 0), training=False).numpy()
            pred_id = int(clf.predict(feat)[0])
            st.success(f"Predicted: **{class_names[pred_id]}**")

            # Top-5 if available
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(feat)[0]
                top5 = np.argsort(probs)[::-1][:5]
                st.write("Top-5:")
                for k in top5:
                    st.write(f"- {class_names[k]} — {probs[k]:.2%}")
