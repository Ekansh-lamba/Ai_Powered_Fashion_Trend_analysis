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

# Optional viz
import plotly.express as px

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

# ---------- Page Config ----------
st.set_page_config(
    page_title="Fashion Trend Analyser",
    layout="wide",
    page_icon="🧵",
)

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

# ---------- Utilities ----------
CAT_COL_MAP = {
    "product_group": "product_group_name",
    "colour_group": "colour_group_name",
    "graphical_appearance": "graphical_appearance_name",
}

def guard_artifacts():
    if not ART.exists():
        st.error("`artifacts/` folder not found. Run your feature extraction & weekly trend scripts first.")
        st.stop()

def to_date(d) -> pd.Timestamp:
    return pd.to_datetime(d, errors="coerce")

# ---------- Tab renderers ----------
def render_trends_tab(weekly: dict):
    st.header("📈 Trends")
    st.caption("Explore weekly sales trends by category with interactive filtering and summaries.")

    # --- Sidebar Controls ---
    with st.sidebar:
        st.markdown("### Trends Filters")
        kind = st.selectbox(
            "Trend type",
            ["product_group", "colour_group", "graphical_appearance"],
            format_func=lambda k: {
                "product_group": "Product Group",
                "colour_group": "Colour Group",
                "graphical_appearance": "Graphical Appearance",
            }[k],
            key="tr_kind",
        )

    df = weekly.get(kind)
    if df is None or df.empty:
        st.warning(
            f"No saved CSV for `{kind}` found in artifacts/, or it is empty. "
            "Run the chunked aggregation to generate it."
        )
        return

    df = df.copy()
    df["week"] = to_date(df["week"])

    cat_col = CAT_COL_MAP[kind]
    if cat_col not in df.columns:
        st.error(f"Expected column `{cat_col}` not found in the CSV. Found columns: {list(df.columns)}")
        return

    # Compute top categories overall
    totals_all = (
        df.groupby(cat_col)["sales_count"]
        .sum()
        .sort_values(ascending=False)
    )

    # --- Sidebar (continued) ---
    with st.sidebar:
        topN = st.slider("Top N categories", 3, 12, 5, key="tr_topn")
        top_cats = totals_all.head(topN).index.tolist()

        dff = df[df[cat_col].isin(top_cats)].copy()
        if dff.empty:
            st.info("No rows match the selected categories.")
            return

        # Date range - must be Python date objects
        min_w = pd.to_datetime(dff["week"].min()).to_pydatetime().date()
        max_w = pd.to_datetime(dff["week"].max()).to_pydatetime().date()
        start_date, end_date = st.slider(
            "Date range",
            value=(min_w, max_w),
            min_value=min_w,
            max_value=max_w,
            key="tr_dates",
        )

    mask = (dff["week"].dt.date >= start_date) & (dff["week"].dt.date <= end_date)
    dff = dff[mask]
    if dff.empty:
        st.info("No data in the selected date range.")
        return

    # Summaries
    range_total = int(dff["sales_count"].sum())
    top_cat_in_range = (
        dff.groupby(cat_col)["sales_count"].sum().sort_values(ascending=False).index[0]
    )
    weeks_in_range = dff["week"].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales (Selected Range)", f"{range_total:,}")
    c2.metric("Top Performing Category", top_cat_in_range)
    c3.metric("Weeks in Range", f"{weeks_in_range}")

    # Pivot for plot
    pivot = (
        dff.pivot(index="week", columns=cat_col, values="sales_count")
        .sort_index()
        .fillna(0)
    )

    # Plotly (dark)
    melted = pivot.reset_index().melt(id_vars="week", var_name="category", value_name="sales")
    fig = px.line(
        melted,
        x="week",
        y="sales",
        color="category",
        markers=True,
        template="plotly_dark",
        title="Weekly Sales by Category",
    )
    fig.update_layout(
        legend_title_text="Category",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Data Preview")
    st.caption("Last 10 rows of the weekly pivot table:")
    st.dataframe(pivot.tail(10))

    st.markdown("#### Totals in Selected Range")
    totals = dff.groupby(cat_col)["sales_count"].sum().sort_values(ascending=False)
    st.dataframe(totals.to_frame("sales_count"))

def render_similarity_tab(meta: pd.DataFrame, X_img: np.memmap):
    st.header("🖼️ Similarity Search")
    st.caption("Find visually similar items using image embeddings and cosine similarity.")

    base, preprocess = load_backbone()
    nn = build_nn_index(X_img)

    # --- Sidebar Controls ---
    with st.sidebar:
        st.markdown("### Similarity Search")
        uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="sim_upload")
        topk = st.slider("Neighbors (k)", 3, 12, 6, key="sim_k")

    if not uploaded:
        st.info("Upload an image in the sidebar to start the search.")
        return

    # Show query
    img = Image.open(uploaded)
    st.image(img, caption="Query", width=300)

    # Inference with spinner
    with st.spinner("Analyzing..."):
        import tensorflow as tf
        x = preprocess(img)
        feat = base(tf.expand_dims(x, 0), training=False).numpy()
        feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
        dists, idxs = nn.kneighbors(feat, n_neighbors=topk)
        # Show a couple of useful columns if present
        keep_cols = [c for c in ["image_path", "product_group_name", "colour_group_name", "graphical_appearance_name"] if c in meta.columns]
        res = meta.iloc[idxs[0]][keep_cols].assign(distance=dists[0])

    st.markdown("#### Nearest Matches")
    # Cards grid
    cols = st.columns(min(topk, 6))
    for j, (_, row) in enumerate(res.iterrows()):
        with cols[j % len(cols)]:
            with st.container(border=True):
                p = Path(row["image_path"]) if "image_path" in row else None
                if p is not None and p.exists():
                    st.image(str(p), use_column_width=True)
                cap_bits = []
                if "product_group_name" in row: cap_bits.append(f"**{row['product_group_name']}**")
                if "colour_group_name" in row: cap_bits.append(f"{row['colour_group_name']}")
                if "graphical_appearance_name" in row: cap_bits.append(f"{row['graphical_appearance_name']}")
                st.caption(" • ".join(cap_bits) if cap_bits else "Match")
                st.caption(f"Cosine distance: {row['distance']:.3f}")

    st.markdown("#### Results Table")
    st.dataframe(res)

def render_predict_tab(clf, class_names):
    st.header("🔮 Prediction")
    st.caption("Classify a product image using your trained classifier.")

    if clf is None or class_names is None:
        st.info("Classifier/classes not found. Save them first (`logreg_img.pkl`, `classes.json` in `artifacts/`).")
        return

    base, preprocess = load_backbone()

    # --- Sidebar Controls ---
    with st.sidebar:
        st.markdown("### Prediction")
        f = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="pred_upload")

    if not f:
        st.info("Upload an image in the sidebar to get a prediction.")
        return

    img = Image.open(f)
    st.image(img, caption="Input", width=300)

    with st.spinner("Analyzing..."):
        import tensorflow as tf
        x = preprocess(img)
        feat = base(tf.expand_dims(x, 0), training=False).numpy()
        pred_id = int(clf.predict(feat)[0])

    with st.container(border=True):
        st.success(f"**Predicted:** {class_names[pred_id]}")

    # Top-5 probabilities (if available)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(feat)[0]
        top5 = np.argsort(probs)[::-1][:5]
        st.markdown("#### Top-5 Probabilities")
        prob_df = pd.DataFrame({
            "class": [class_names[k] for k in top5],
            "probability": [float(probs[k]) for k in top5],
        })
        st.dataframe(prob_df)

# ---------- App ----------
def main():
    guard_artifacts()

    # Load artifacts up-front
    weekly = load_weekly_csvs()
    meta, X_img = load_meta_features()
    clf, class_names = load_classifier()

    # Title & tabs
    st.title("🧵 Fashion Trend Analyser")
    st.caption("An interactive dashboard for weekly trend analysis, visual similarity search, and product group prediction.")

    tab_trend, tab_sim, tab_pred = st.tabs(["Trends", "Similarity Search", "Predict"])

    with tab_trend:
        render_trends_tab(weekly)

    with tab_sim:
        render_similarity_tab(meta, X_img)

    with tab_pred:
        render_predict_tab(clf, class_names)

if __name__ == "__main__":
    main()
