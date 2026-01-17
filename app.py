import streamlit as st
import pandas as pd
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Flood Intelligence Platform", layout="wide", page_icon="🌊")

# ================= LOAD STYLES =================
if os.path.exists("style.css"):
    st.markdown(f"<style>{open('style.css').read()}</style>", unsafe_allow_html=True)
if os.path.exists("header.html"):
    st.markdown(open("header.html").read(), unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("tamilnadu_flood_dataset_with_hydro_params.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = [
        "year","month","district","rainfall_mm","duration_days","main_cause","severity",
        "latitude","longitude","runoff_mm","flood_depth_m","risk_zone","land_use","curve_number"
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    for c in ["rainfall_mm","duration_days","month","year","runoff_mm","flood_depth_m","curve_number"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["district"] = df["district"].astype(str).fillna("Unknown")
    df["main_cause"] = df["main_cause"].astype(str).fillna("Unknown")
    df["land_use"] = df["land_use"].astype(str).fillna("Unknown")
    df["severity"] = df["severity"].astype(str).str.lower().str.strip()
    df["risk_zone"] = df["risk_zone"].astype(str).str.lower().str.strip()

    # FIX: binary target explicitly
    severity_map = {"low": 0, "medium": 1, "high": 1}
    df["target"] = df["severity"].map(severity_map)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    return df

df = load_data()

# ================= SIDEBAR =================
st.sidebar.title("🔧 Controls")

with st.sidebar.expander("Location", expanded=True):
    sel_districts = st.multiselect("Districts", sorted(df["district"].unique()), default=sorted(df["district"].unique())[:5])

with st.sidebar.expander("Time", expanded=True):
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    yr_range = st.slider("Year Range", yr_min, yr_max, (yr_min, yr_max))

with st.sidebar.expander("Display", expanded=True):
    chart_type = st.radio("Chart Type", ["Rainfall vs Duration", "Severity Scatter"])

# ================= FILTER =================
filtered = df.copy()
filtered = filtered[filtered["district"].isin(sel_districts)]
filtered = filtered[filtered["year"].between(yr_range[0], yr_range[1])]

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["📊 Analytics", "📈 Visuals", "🤖 Prediction"])

# ================= ANALYTICS =================
with tab1:
    st.header("Flood Analytics")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Events", len(filtered))
    c2.metric("High Severity %", f"{filtered['target'].mean()*100:.1f}%")
    c3.metric("Avg Rainfall (mm)", f"{filtered['rainfall_mm'].mean():.1f}")
    c4.metric("Avg Duration (days)", f"{filtered['duration_days'].mean():.1f}")

    st.plotly_chart(px.bar(filtered.groupby("district")["target"].mean().reset_index(),
                           x="district", y="target",
                           title="Severity Ratio by District",
                           color="target", color_continuous_scale="Reds"),
                    use_container_width=True)

    st.plotly_chart(px.line(filtered.groupby("year")["target"].mean().reset_index(),
                            x="year", y="target",
                            title="Severity Trend Over Time",
                            markers=True),
                    use_container_width=True)

# ================= VISUALS =================
with tab2:
    st.header("Visual Analysis")

    if filtered.empty:
        st.warning("No data available for selected filters.")
    else:
        if chart_type == "Rainfall vs Duration":
            fig = px.scatter(filtered, x="rainfall_mm", y="duration_days",
                             size="target", color="target",
                             title="Rainfall vs Duration (Severity)",
                             color_continuous_scale="Reds")
        else:
            fig = px.scatter(filtered, x="year", y="rainfall_mm",
                             color="target",
                             title="Severity Scatter Over Time",
                             color_continuous_scale="Reds")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🗺️ Flood Severity Map")
        map_fig = px.scatter_mapbox(
            filtered,
            lat="latitude", lon="longitude",
            color="target", size="rainfall_mm",
            hover_name="district",
            hover_data=["rainfall_mm","duration_days","main_cause","severity","runoff_mm","flood_depth_m","risk_zone"],
            zoom=6, height=600,
            color_continuous_scale="Reds"
        )
        map_fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(map_fig, use_container_width=True)

# ================= PREDICTION =================
with tab3:
    st.header("Flood Severity Prediction")

    le_d = LabelEncoder()
    le_c = LabelEncoder()
    le_l = LabelEncoder()

    df["district_enc"] = le_d.fit_transform(df["district"])
    df["cause_enc"] = le_c.fit_transform(df["main_cause"])
    df["land_enc"] = le_l.fit_transform(df["land_use"])

    @st.cache_resource
    def train_model(X,y):
        model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, eval_metric="logloss")
        model.fit(X,y)
        return model

    model = train_model(
        df[["district_enc","cause_enc","land_enc","rainfall_mm","duration_days","curve_number","month","runoff_mm","flood_depth_m"]],
        df["target"]
    )

    col1, col2 = st.columns(2)
    d = col1.selectbox("District", le_d.classes_)
    c = col1.selectbox("Main Cause", le_c.classes_)
    l = col1.selectbox("Land Use", le_l.classes_)
    r = col2.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
    dur = col2.number_input("Duration (days)", 1, 30, 5)
    cn = col2.slider("Curve Number", 60, 98, 80)
    m = col2.slider("Month", 1, 12, 6)

    if st.button("Predict"):
        # FIX: use dataset-consistent runoff
        runoff = r * 0.6
        depth = runoff / 1000

        X = [[
            le_d.transform([d])[0],
            le_c.transform([c])[0],
            le_l.transform([l])[0],
            r, dur, cn, m, runoff, depth
        ]]

        probs = model.predict_proba(X)[0]  # FIX
        prob = probs[1]                    # FIX: probability of high/medium

        st.progress(int(prob*100))
        st.metric("Flood Risk Probability", f"{prob*100:.1f}%")

        if prob > 0.5:
            st.error("🔴 High Risk Flood Event")
        else:
            st.success("🟢 Low Risk Flood Event")
