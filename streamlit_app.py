import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from ibm_watson_machine_learning import APIClient

# Check if statsmodels is available
try:
    import statsmodels
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ==========================================
# 1. IBM WATSON AUTHENTICATION
# ==========================================
def load_credentials():
    if os.path.exists('apikey.json'):
        with open('apikey.json', 'r') as f:
            data = json.load(f)
            return data.get("apikey")
    return None

IBM_API_KEY = load_credentials()
WML_CREDENTIALS = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": IBM_API_KEY
}

# --- YOUR SPACE ID FROM SCREENSHOT ---
SPACE_ID = "4cdacfe5-9fe3-4e13-b61c-e9c0f98ce55f"

# ==========================================
# 2. DATA LOADING & KPI CALCULATIONS
# ==========================================
@st.cache_data
def load_and_calculate():
    df = pd.read_csv('us_open_sponsor_rich_dataset.csv')

    # Calculate Summary Statistics
    total_impressions = df['impressions'].sum()
    avg_sentiment = df['sentiment_score'].mean()

    # Calculate Growth (Latest Year vs Previous Year)
    yearly_val = df.groupby('year')['media_value_million_usd'].sum()
    latest_yr = yearly_val.index.max()
    prev_yr = latest_yr - 1
    growth = ((yearly_val[latest_yr] - yearly_val[prev_yr]) / yearly_val[prev_yr]) * 100

    # One-hot encoding for ML
    df_ml = pd.get_dummies(df, columns=['sponsor'], drop_first=True)

    return df, df_ml, total_impressions, avg_sentiment, growth

df_raw, df, total_imp, avg_sent, growth_rate = load_and_calculate()

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="US Open Sponsor Analytics", layout="wide")
st.title("🎾 US Open Sponsor Performance Dashboard")
st.markdown("---")

# KPI Metrics Section
st.header("Executive Summary")
k1, k2, k3 = st.columns(3)
k1.metric("Total Impressions", f"{total_imp:,}")
k2.metric("Average Sentiment Score", f"{avg_sent:.2f}")
k3.metric("YoY Media Value Growth", f"{growth_rate:.2f}%")

# Sidebar Configuration
st.sidebar.header("ML Configuration")
target = 'media_value_million_usd'
features = st.sidebar.multiselect(
    "Select Input Features",
    [c for c in df.columns if c != target],
    default=['cost_million_usd', 'impressions', 'engagements', 'sentiment_score']
)

if not features:
    st.warning("Please select at least one feature to continue.")
    st.stop()

# Machine Learning Logic
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_train)
X_te_scaled = scaler.transform(X_test)

st.header("1. Model Performance")
algo = st.selectbox("Choose Algorithm", ["Random Forest", "Linear Regression"])
model = RandomForestRegressor(n_estimators=100) if algo == "Random Forest" else LinearRegression()
model.fit(X_tr_scaled, y_train)

y_pred = model.predict(X_te_scaled)
score = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric(f"{algo} R² Score", f"{score:.4f}")
col2.metric(f"{algo} MAE", f"{mae:.4f}")

# Visualization Section
st.header("2. Data Visualization")

if HAS_STATSMODELS:
    fig = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual Media Value ($M)', 'y': 'Predicted Media Value ($M)'},
        title=f"Actual vs Predicted Comparison ({algo})",
        template="plotly_white",
        trendline="ols"
    )
else:
    st.info("💡 Install `statsmodels` (`pip install statsmodels`) to enable the OLS trendline.")
    fig = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual Media Value ($M)', 'y': 'Predicted Media Value ($M)'},
        title=f"Actual vs Predicted Comparison ({algo})",
        template="plotly_white"
    )

# Add ideal fit line
fig.add_trace(go.Scatter(
    x=[y.min(), y.max()],
    y=[y.min(), y.max()],
    mode='lines',
    name='Ideal Fit',
    line=dict(color='red', dash='dash')
))

st.plotly_chart(fig, use_container_width=True)

# IBM Cloud Synchronization
st.divider()
st.header("3. Cloud Asset Synchronization")
st.info("Upload this specific model to your IBM Deployment Space.")

if st.button("🚀 Sync Model to IBM Watson ML"):
    if not IBM_API_KEY:
        st.error("API Key missing. Please check your apikey.json file.")
    else:
        try:
            client = APIClient(WML_CREDENTIALS)
            client.set.default_space(SPACE_ID)

            # Use software specification: runtime-24.1-py3.11
            spec_id = client.software_specifications.get_id_by_name("runtime-25.1-py3.11")

            meta_props = {
                client.repository.ModelMetaNames.NAME: f"US_Open_Model_{algo}",
                client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: spec_id,
                client.repository.ModelMetaNames.TYPE: "scikit-learn_1.3",
                client.repository.ModelMetaNames.CUSTOM: {
                    "R2_Score": float(score),
                    "Total_Impressions": int(total_imp)
                }
            }

            # Perform storage
            details = client.repository.store_model(model=model, meta_props=meta_props)
            st.success("Successfully uploaded! Check your 'Assets' tab in IBM Cloud.")
            st.code(f"Model ID: {details['metadata']['id']}")
            st.balloons()
        except Exception as e:
            st.error(f"Synchronization failed: {e}")
