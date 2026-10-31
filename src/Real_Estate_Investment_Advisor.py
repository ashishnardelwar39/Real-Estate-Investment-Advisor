"""
Real Estate Investment Advisor - Streamlit app
------------------------------------------------
Run locally :  streamlit run app.py
Deploy      :  push app.py, requirements.txt and india_housing_prices.csv to GitHub,
               then deploy the repo on Streamlit Community Cloud.

On first run the app trains two models and caches them in ./models:
  1. Good_Investment classifier  (5-rule score >= 3 -> 1)
  2. Future_Price_5Y regressor   (price after 5 years of city-wise growth)
"""

from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Config (relative paths -> works on any machine / Streamlit Cloud)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "india_housing_prices.csv"
MODEL_DIR = BASE_DIR / "models"
BUNDLE_PATH = MODEL_DIR / "bundle.joblib"

CITY_GROWTH = {"Mumbai": 0.08, "Pune": 0.07, "Nagpur": 0.06}
DEFAULT_GROWTH = 0.05
MAX_TRAIN_ROWS = 50_000  # keeps training fast and the model file small

BASE_NUM = ["BHK", "Size_in_SqFt", "Year_Built", "Floor_No", "Total_Floors",
            "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
            "Amenities_Count", "Infra_Score"]
PRICE_NUM = ["Price_in_Lakhs", "Price_per_SqFt",
             "PPSF_vs_City_Median", "Price_vs_City_Median"]
CAT_COLS = ["State", "City", "Locality", "Property_Type", "Furnished_Status",
            "Parking_Space", "Security", "Facing", "Owner_Type",
            "Availability_Status", "Public_Transport_Accessibility"]

CLF_NUM = BASE_NUM + PRICE_NUM   # classifier may see price (the rules use it)
REG_NUM = BASE_NUM               # regressor must NOT see price (would be leakage)


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
    if "Amenities_Count" not in df.columns:
        df["Amenities_Count"] = (
            df["Amenities"].fillna("").astype(str)
            .apply(lambda s: len([a for a in s.split(",") if a.strip()]))
        )
    df["Infra_Score"] = (df["Nearby_Schools"] / 10 + df["Nearby_Hospitals"] / 10) / 2 * 10
    df["Growth_Rate"] = df["City"].map(CITY_GROWTH).fillna(DEFAULT_GROWTH)
    df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** 5
    return df


def add_city_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    df["City_Median_PPSF"] = df["City"].map(stats["ppsf"]).fillna(stats["ppsf_all"])
    df["City_Median_Price"] = df["City"].map(stats["price"]).fillna(stats["price_all"])
    df["PPSF_vs_City_Median"] = df["Price_per_SqFt"] / df["City_Median_PPSF"]
    df["Price_vs_City_Median"] = df["Price_in_Lakhs"] / df["City_Median_Price"]
    return df


def rule_checks(df: pd.DataFrame, infra_median: float) -> pd.DataFrame:
    return pd.DataFrame({
        "Price/SqFt <= city median": df["Price_per_SqFt"] <= df["City_Median_PPSF"],
        "Price <= city median": df["Price_in_Lakhs"] <= df["City_Median_Price"],
        "BHK >= 3": df["BHK"] >= 3,
        "Ready to move": df["Availability_Status"].astype(str).str.strip().str.lower() == "ready_to_move",
        "Infra score >= median": df["Infra_Score"] >= infra_median,
    })


def make_pipeline(num_cols, cat_cols, model):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([("prep", prep), ("model", model)])


# ----------------------------------------------------------------------------
# Data + training
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer).drop_duplicates()
    df = df.dropna(subset=["Price_in_Lakhs", "Size_in_SqFt", "City"])
    df = df.drop(columns=["ID"], errors="ignore")
    if "Age_of_Property" not in df.columns and "Year_Built" in df.columns:
        df["Age_of_Property"] = date.today().year - df["Year_Built"]
    df = add_base_features(df)
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def train_bundle(df: pd.DataFrame) -> dict:
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    stats = {
        "ppsf": df.groupby("City")["Price_per_SqFt"].median().to_dict(),
        "price": df.groupby("City")["Price_in_Lakhs"].median().to_dict(),
        "ppsf_all": float(df["Price_per_SqFt"].median()),
        "price_all": float(df["Price_in_Lakhs"].median()),
        "infra_median": float(df["Infra_Score"].median()),
    }
    df = add_city_features(df, stats)

    # Target 1: Good_Investment = at least 3 of 5 rules satisfied
    score = rule_checks(df, stats["infra_median"]).astype(int).sum(axis=1)
    df["Good_Investment"] = (score >= 3).astype(int)

    if len(df) > MAX_TRAIN_ROWS:
        df = df.sample(MAX_TRAIN_ROWS, random_state=42)

    # ---- classifier
    Xc = df[CLF_NUM + cat_cols]
    yc = df["Good_Investment"]
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
    clf = make_pipeline(CLF_NUM, cat_cols, RandomForestClassifier(
        n_estimators=80, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=42))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    clf_metrics = {"Accuracy": accuracy_score(yte, pred),
                   "F1 (weighted)": f1_score(yte, pred, average="weighted")}

    # ---- regressor (Target 2: Future_Price_5Y, price columns excluded)
    Xr = df[REG_NUM + cat_cols]
    yr = df["Future_Price_5Y"]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg = make_pipeline(REG_NUM, cat_cols, RandomForestRegressor(
        n_estimators=80, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=42))
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    reg_metrics = {"MAE (Lakhs)": mean_absolute_error(yte, pred),
                   "RMSE (Lakhs)": float(np.sqrt(mean_squared_error(yte, pred))),
                   "R2": r2_score(yte, pred)}

    return {"clf": clf, "reg": reg, "stats": stats, "cat_cols": cat_cols,
            "clf_metrics": clf_metrics, "reg_metrics": reg_metrics}


@st.cache_resource(show_spinner="Training models (first run only, ~1-2 min)...")
def get_bundle(_df: pd.DataFrame, data_signature: int) -> dict:
    if BUNDLE_PATH.exists():
        try:
            return joblib.load(BUNDLE_PATH)
        except Exception:
            pass  # corrupt / incompatible version -> retrain
    bundle = train_bundle(_df)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    return bundle


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")
st.title("🏠 Real Estate Investment Advisor")
st.caption("Predicts whether a property is a good investment and its estimated price after 5 years.")

# ---- data source
source = None
if DATA_PATH.exists():
    source = DATA_PATH
else:
    up = st.sidebar.file_uploader("Upload india_housing_prices.csv", type="csv")
    if up is not None:
        source = up
if source is None:
    st.warning("`india_housing_prices.csv` not found next to app.py. Place it there or upload it in the sidebar.")
    st.stop()

df = load_data(source)
bundle = get_bundle(df, len(df))
stats, cat_cols = bundle["stats"], bundle["cat_cols"]

tab_pred, tab_eda, tab_model = st.tabs(["🔮 Predict", "📊 Market insights", "🧪 Model performance"])

# ============================ Predict ============================
with tab_pred:
    city_state = df.groupby("City")["State"].agg(lambda s: s.mode().iat[0]).to_dict()
    amenity_opts = sorted({a.strip() for s in df["Amenities"].dropna().astype(str).unique()
                           for a in s.split(",") if a.strip()}) if "Amenities" in df.columns else []

    def opts(col):
        return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

    c1, c2, c3 = st.columns(3)
    with c1:
        city = st.selectbox("City", sorted(df["City"].unique()))
        localities = sorted(df.loc[df["City"] == city, "Locality"].unique()) if "Locality" in df.columns else []
        locality = st.selectbox("Locality", localities) if localities else None
        ptype = st.selectbox("Property type", opts("Property_Type")) if opts("Property_Type") else None
        bhk = st.number_input("BHK", 1, 10, 3)
        size = st.number_input("Size (SqFt)", 100, 20000, 1200, step=50)
        price = st.number_input("Asking price (Lakhs)", 1.0, 100000.0, float(round(stats["price"].get(city, stats["price_all"]), 1)), step=1.0)
    with c2:
        year_built = st.number_input("Year built", 1950, date.today().year + 3, 2015)
        floor_no = st.number_input("Floor no.", 0, 100, 2)
        total_floors = st.number_input("Total floors", 1, 100, 10)
        schools = st.slider("Nearby schools", 0, 10, 5)
        hospitals = st.slider("Nearby hospitals", 0, 10, 5)
        amenities = st.multiselect("Amenities", amenity_opts)
    with c3:
        furnished = st.selectbox("Furnishing", opts("Furnished_Status")) if opts("Furnished_Status") else None
        parking = st.selectbox("Parking space", opts("Parking_Space")) if opts("Parking_Space") else None
        security = st.selectbox("Security", opts("Security")) if opts("Security") else None
        facing = st.selectbox("Facing", opts("Facing")) if opts("Facing") else None
        owner = st.selectbox("Owner type", opts("Owner_Type")) if opts("Owner_Type") else None
        avail = st.selectbox("Availability", opts("Availability_Status")) if opts("Availability_Status") else None
        transport = st.selectbox("Public transport accessibility", opts("Public_Transport_Accessibility")) if opts("Public_Transport_Accessibility") else None

    if st.button("Analyse property", type="primary"):
        row = {
            "State": city_state.get(city), "City": city, "Locality": locality,
            "Property_Type": ptype, "Furnished_Status": furnished, "Parking_Space": parking,
            "Security": security, "Facing": facing, "Owner_Type": owner,
            "Availability_Status": avail, "Public_Transport_Accessibility": transport,
            "BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": price,
            "Year_Built": year_built, "Floor_No": floor_no, "Total_Floors": total_floors,
            "Age_of_Property": max(date.today().year - year_built, 0),
            "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
            "Amenities_Count": len(amenities),
        }
        X = pd.DataFrame([row])
        X["Price_per_SqFt"] = X["Price_in_Lakhs"] * 1e5 / X["Size_in_SqFt"]
        X["Infra_Score"] = (X["Nearby_Schools"] / 10 + X["Nearby_Hospitals"] / 10) / 2 * 10
        X = add_city_features(X, stats)
        for c in cat_cols:
            X[c] = X[c].astype(str)

        proba = float(bundle["clf"].predict_proba(X[CLF_NUM + cat_cols])[0][1])
        future_model = float(bundle["reg"].predict(X[REG_NUM + cat_cols])[0])
        growth = CITY_GROWTH.get(city, DEFAULT_GROWTH)
        future_formula = price * (1 + growth) ** 5

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Investment verdict", "✅ Good" if proba >= 0.5 else "⚠️ Not ideal", f"{proba:.0%} confidence")
        m2.metric(f"Price in 5 yrs @ {growth:.0%}/yr growth", f"₹ {future_formula:,.1f} L",
                  f"{future_formula - price:+,.1f} L vs today")
        m3.metric("Model's market-value estimate (5Y)", f"₹ {future_model:,.1f} L")

        st.subheader("Why? - the 5 investment rules")
        checks = rule_checks(X, stats["infra_median"]).iloc[0]
        for rule, ok in checks.items():
            st.write(("✔️ " if ok else "❌ ") + rule)
        st.caption(f"Rules passed: {int(checks.sum())}/5 (a property is 'good' when at least 3 pass). "
                   f"City median: ₹{stats['ppsf'].get(city, stats['ppsf_all']):,.0f}/sqft, "
                   f"₹{stats['price'].get(city, stats['price_all']):,.1f} L.")
        st.info("Indicative only - not financial advice.")

# ============================ EDA ============================
with tab_eda:
    plot_df = df.sample(min(len(df), 20000), random_state=1)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots()
        sns.histplot(plot_df["Price_in_Lakhs"], kde=True, ax=ax)
        ax.set(title="Distribution of property prices", xlabel="Price (Lakhs)")
        st.pyplot(fig)
    with b:
        fig, ax = plt.subplots()
        df.groupby("City")["Price_per_SqFt"].mean().sort_values().plot(kind="barh", ax=ax)
        ax.set(title="Average price per SqFt by city", xlabel="₹ / SqFt")
        st.pyplot(fig)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots()
        sns.scatterplot(data=plot_df, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=0.3, ax=ax)
        ax.set_title(f"Size vs price (corr = {df['Size_in_SqFt'].corr(df['Price_in_Lakhs']):.2f})")
        st.pyplot(fig)
    with b:
        fig, ax = plt.subplots()
        pd.crosstab(df["City"], df["BHK"]).plot(kind="bar", stacked=True, ax=ax)
        ax.set(title="BHK mix by city", ylabel="Properties")
        st.pyplot(fig)
    plt.close("all")

# ============================ Model performance ============================
with tab_model:
    a, b = st.columns(2)
    with a:
        st.subheader("Good-investment classifier")
        st.table(pd.Series(bundle["clf_metrics"], name="Score").round(4))
        st.caption("Random Forest. The target is built from rules on price, BHK, availability and infrastructure, "
                   "so high accuracy is expected.")
    with b:
        st.subheader("Future-price regressor")
        st.table(pd.Series(bundle["reg_metrics"], name="Score").round(4))
        st.caption("Random Forest. Price columns are deliberately excluded from its inputs to avoid target leakage.")
    st.caption("To retrain, delete the `models/` folder and restart the app.")
