import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import numpy as np
from blockchain_logger import BlackBoxLedger # <--- NEW IMPORT

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PredMaint Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1b. SECURITY AUDIT (Run on load)
# ==========================================
ledger = BlackBoxLedger()
is_secure, security_msg = ledger.validate_integrity()

# ==========================================
# 2. MODERN LIGHT THEME CSS (Card Style)
# ==========================================
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 20px;
    }
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    if not os.path.exists("test_features.csv"): return None
    return pd.read_csv("test_features.csv")

@st.cache_resource
def load_model():
    if not os.path.exists("isolation_forest_model.joblib"): return None
    return joblib.load("isolation_forest_model.joblib")

df_test = load_data()
model = load_model()

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3061/3061341.png", width=50)
st.sidebar.title("Fleet Manager")
st.sidebar.markdown("---")

if df_test is not None:
    engine_ids = df_test['unit_number'].unique()
    selected_engine = st.sidebar.selectbox("Select Asset ID:", engine_ids)
    st.sidebar.info(f"Viewing telemetry for Unit #{selected_engine}")
    engine_data = df_test[df_test['unit_number'] == selected_engine].copy()
else:
    st.error("Data missing. Please run the pipeline.")
    st.stop()

# ==========================================
# 5. DATA PROCESSING (Health Score)
# ==========================================
feature_cols = [c for c in engine_data.columns if c not in ['unit_number', 'time_in_cycles', 'is_anomaly']]
X_engine = engine_data[feature_cols]

raw_scores = model.decision_function(X_engine)
engine_data['health_index'] = np.clip(((raw_scores + 0.15) / 0.3) * 100, 0, 100)
engine_data['status'] = np.where(engine_data['health_index'] > 50, 'Healthy', 'Critical')

current_health = engine_data['health_index'].iloc[-1]
total_cycles = len(engine_data)
health_delta = current_health - engine_data['health_index'].iloc[-2] if total_cycles > 1 else 0

# ==========================================
# 6. DASHBOARD LAYOUT
# ==========================================

# HEADER
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title(f"Engine Unit #{selected_engine} Overview")
    
    # <--- SECURITY BADGE --->
    if is_secure:
        st.caption(f"🔒 BLOCKCHAIN SECURED | Ledger Height: {len(ledger.chain)} blocks | Verification: PASSED")
    else:
        st.markdown(f"SECURITY ALERT: {security_msg}")
        st.error("System Integrity Compromised. Maintenance logs have been tampered with.")
        st.stop() # Stops rendering data if hacked

with col_status:
    status_color = "green" if current_health > 50 else "red"
    st.markdown(f"""
        <div style="text-align: right; padding: 10px;">
            <span style="background-color:{status_color}; color:white; padding: 5px 10px; border-radius: 5px; font-weight:bold;">
                STATUS: { "OPTIMAL" if current_health > 50 else "ATTENTION REQUIRED" }
            </span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ROW 1: METRIC CARDS
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(label="Health Score", value=f"{current_health:.1f}%", delta=f"{health_delta:.2f}%")
with m2:
    st.metric(label="Operating Cycles", value=f"{total_cycles}")
with m3:
    risk_level = "Low" if current_health > 70 else ("Medium" if current_health > 50 else "High")
    st.metric(label="Risk Assessment", value=risk_level, delta_color="inverse")
with m4:
    rul = max(0, 150 - total_cycles)
    st.metric(label="Est. RUL (Cycles)", value=f"{rul}")

st.markdown("<br>", unsafe_allow_html=True)

# ROW 2: MAIN CHARTS
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader(" Health Degradation Curve")
    fig_health = px.area(
        engine_data, 
        x="time_in_cycles", 
        y="health_index",
        labels={"health_index": "Health Index (%)", "time_in_cycles": "Cycle Count"},
        template="plotly_white", 
        height=350
    )
    fill_color = 'rgba(46, 204, 113, 0.3)' if current_health > 50 else 'rgba(231, 76, 60, 0.3)'
    line_color = '#2ecc71' if current_health > 50 else '#e74c3c'
    
    fig_health.update_traces(line_color=line_color, fillcolor=fill_color)
    fig_health.add_hline(y=50, line_dash="dot", line_color="red", annotation_text="Threshold")
    st.plotly_chart(fig_health, use_container_width=True)

with c2:
    st.subheader("📡 Recent Telemetry")
    recent_df = engine_data[['time_in_cycles', 'health_index', 'status']].tail(8).sort_values(by='time_in_cycles', ascending=False)
    st.dataframe(
        recent_df, 
        hide_index=True, 
        column_config={
            "health_index": st.column_config.ProgressColumn("Health", format="%.1f%%", min_value=0, max_value=100),
            "status": st.column_config.TextColumn("Status")
        },
        use_container_width=True,
        height=350
    )

# ROW 3: DETAILED ANALYSIS TABS
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["🔍 Deep Dive: Sensor Correlation", "🔗 Immutable Ledger Inspector"])

with tab1:
    s_opts = [c for c in feature_cols if 'sensor' in c]
    sel_sensors = st.multiselect("Compare Sensors:", s_opts, default=["sensor_2", "sensor_3", "sensor_4"])
    if sel_sensors:
        fig_sens = px.line(engine_data, x="time_in_cycles", y=sel_sensors, template="plotly_white", height=300)
        st.plotly_chart(fig_sens, use_container_width=True)

with tab2:
    st.markdown("### ⛓️ Blockchain Data Integrity Verification")
    if is_secure:
        st.success(f" CHAIN VERIFIED. All {len(ledger.chain)} blocks are cryptographically valid.")
        
        # Display the last 5 blocks in a visual format
        st.markdown("**Most Recent Blocks:**")
        recents = ledger.chain[-5:]
        for b in reversed(recents):
            with st.expander(f"Block #{b.index} | Status: {b.data['status']} | Hash: {b.hash[:15]}..."):
                st.json(b.__dict__)
    else:
        st.error("Chain Validation Failed.")

# FOOTER
st.markdown("---")
st.caption("Predictive Maintenance Dashboard v2.0 (Enterprise Edition) | © 2026 Pooja Kiran")