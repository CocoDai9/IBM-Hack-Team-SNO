import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import requests
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
    # 优先读 Streamlit Secrets（线上 Streamlit Cloud 环境）
    try:
        return st.secrets["ibm"]["apikey"]
    except:
        # 本地开发时读 apikey.json
        if os.path.exists('apikey.json'):
            with open('apikey.json', 'r') as f:
                return json.load(f).get("apikey")
    return None

IBM_API_KEY = load_credentials()
WML_CREDENTIALS = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": IBM_API_KEY
}

SPACE_ID        = "4cdacfe5-9fe3-4e13-b61c-e9c0f98ce55f"
DEPLOYMENT_ID   = "019dd049-95e0-7656-96c5-152889de3be6"
PUBLIC_ENDPOINT = (
    "https://us-south.ml.cloud.ibm.com/ml/v4/deployments/"
    f"{DEPLOYMENT_ID}/predictions?version=2021-05-01"
)

# ==========================================
# 2. DATA LOADING & KPI CALCULATIONS
# ==========================================
@st.cache_data
def load_and_calculate():
    df = pd.read_csv('us_open_sponsor_rich_dataset.csv')

    total_impressions = df['impressions'].sum()
    avg_sentiment     = df['sentiment_score'].mean()

    yearly_val = df.groupby('year')['media_value_million_usd'].sum()
    latest_yr  = yearly_val.index.max()
    prev_yr    = latest_yr - 1
    growth     = ((yearly_val[latest_yr] - yearly_val[prev_yr]) / yearly_val[prev_yr]) * 100

    df_ml = pd.get_dummies(df, columns=['sponsor'], drop_first=True)
    return df, df_ml, total_impressions, avg_sentiment, growth

df_raw, df, total_imp, avg_sent, growth_rate = load_and_calculate()

# ==========================================
# 3. IBM IAM TOKEN HELPER
# ==========================================
@st.cache_data(ttl=3500)
def get_iam_token(api_key):
    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    return resp.json().get("access_token")

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="IBM-Hack-Team-SNO | US Open Sponsor Analytics", layout="wide")
st.title("US Open Sponsor Performance Dashboard")
st.caption("Repo: [IBM-Hack-Team-SNO](https://github.com/CocoDai9/IBM-Hack-Team-SNO)")
st.markdown("---")

# KPI Metrics
st.header("Executive Summary")
k1, k2, k3 = st.columns(3)
k1.metric("Total Impressions",       f"{total_imp:,}")
k2.metric("Average Sentiment Score", f"{avg_sent:.2f}")
k3.metric("YoY Media Value Growth",  f"{growth_rate:.2f}%")

# Sidebar
st.sidebar.header("ML Configuration")
target   = 'media_value_million_usd'
features = st.sidebar.multiselect(
    "Select Input Features",
    [c for c in df.columns if c != target],
    default=['cost_million_usd', 'impressions', 'engagements', 'sentiment_score']
)

if not features:
    st.warning("Please select at least one feature to continue.")
    st.stop()

# ML pipeline
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler      = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_train)
X_te_scaled = scaler.transform(X_test)

st.header("1. Model Performance")
algo  = st.selectbox("Choose Algorithm", ["Random Forest", "Linear Regression"])
model = RandomForestRegressor(n_estimators=100) if algo == "Random Forest" else LinearRegression()
model.fit(X_tr_scaled, y_train)

y_pred = model.predict(X_te_scaled)
score  = r2_score(y_test, y_pred)
mae    = mean_absolute_error(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric(f"{algo} R² Score", f"{score:.4f}")
col2.metric(f"{algo} MAE",      f"{mae:.4f}")

# Visualization
st.header("2. Data Visualization")
if HAS_STATSMODELS:
    fig = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual Media Value ($M)', 'y': 'Predicted Media Value ($M)'},
        title=f"Actual vs Predicted ({algo})",
        template="plotly_white",
        trendline="ols"
    )
else:
    st.info("💡 Install `statsmodels` to enable OLS trendline.")
    fig = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual Media Value ($M)', 'y': 'Predicted Media Value ($M)'},
        title=f"Actual vs Predicted ({algo})",
        template="plotly_white"
    )

fig.add_trace(go.Scatter(
    x=[y.min(), y.max()], y=[y.min(), y.max()],
    mode='lines', name='Ideal Fit',
    line=dict(color='red', dash='dash')
))
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. LIVE PREDICTION via IBM ENDPOINT
# ==========================================
st.divider()
st.header("3. Live Prediction via IBM Watson")
st.info(f"Calling deployment: `{DEPLOYMENT_ID}`")

with st.form("prediction_form"):
    st.subheader("Enter input values")
    input_cols = st.columns(len(features))
    input_vals = []
    for i, feat in enumerate(features):
        val = input_cols[i].number_input(feat, value=float(X[feat].mean()), format="%.4f")
        input_vals.append(val)
    submitted = st.form_submit_button("Predict")

if submitted:
    if not IBM_API_KEY:
        st.error("API Key missing. Please check apikey.json.")
    else:
        try:
            token   = get_iam_token(IBM_API_KEY)
            payload = {"input_data": [{"values": [input_vals]}]}
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json"
            }
            response = requests.post(PUBLIC_ENDPOINT, json=payload, headers=headers)
            result   = response.json()
            
            prediction = result['predictions'][0]['values'][0][0]
            st.success(f"Predicted Media Value: **${prediction:.2f}M**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ==========================================
# 6. CLOUD ASSET SYNCHRONIZATION
# ==========================================
st.divider()
st.header("4. Cloud Asset Synchronization")
st.info("Upload & deploy this model to IBM Watson ML (Team SNO space).")

if st.button("Sync Model to IBM Watson ML"):
    if not IBM_API_KEY:
        st.error("API Key missing. Please check apikey.json.")
    else:
        try:
            client = APIClient(WML_CREDENTIALS)
            client.set.default_space(SPACE_ID)

            spec_id = client.software_specifications.get_id_by_name("runtime-24.1-py3.11")

            meta_props = {
                client.repository.ModelMetaNames.NAME: f"US_Open_Model_{algo}",
                client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: spec_id,
                client.repository.ModelMetaNames.TYPE: "scikit-learn_1.3",
                client.repository.ModelMetaNames.CUSTOM: {
                    "R2_Score":          float(score),
                    "Total_Impressions": int(total_imp)
                }
            }

            details  = client.repository.store_model(model=model, meta_props=meta_props)
            model_id = details['metadata']['id']

            deploy_meta = {
                client.deployments.ConfigurationMetaNames.NAME:   f"US_Open_Deploy_{algo}",
                client.deployments.ConfigurationMetaNames.ONLINE: {}
            }
            deployment = client.deployments.create(artifact_uid=model_id, meta_props=deploy_meta)
            deploy_id  = deployment['metadata']['id']

            st.success("Model uploaded and deployed!")
            st.code(f"Model ID:      {model_id}\nDeployment ID: {deploy_id}")
            st.balloons()
        except Exception as e:
            st.error(f"Synchronization failed: {e}")