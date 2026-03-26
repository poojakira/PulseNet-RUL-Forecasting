"""
Enhanced Streamlit Dashboard — real-time anomaly monitoring, sensor trends,
blockchain status, system metrics, and multi-engine support.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from pulsenet.security.blockchain import BlackBoxLedger  # noqa: E402
from pulsenet.models.isolation_forest import IsolationForestModel  # noqa: E402

# ===========================================================
# PAGE CONFIG
# ===========================================================
st.set_page_config(
    page_title="PulseNet Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================
# THEME: NVIDIA Production Dark Mode
# ===========================================================
st.markdown(
    """
<style>
    /* Main background: Deep dark gray/black */
    .stApp { background: #0f1115; }
    
    /* Sidebar: Slightly lighter gray */
    [data-testid="stSidebar"] { background: #1a1c23; border-right: 1px solid #2d3038; }
    
    /* Typography: NVIDIA Green (#76b900) for headers */
    h1, h2, h3 { color: #76b900 !important; font-family: 'Inter', 'Segoe UI', sans-serif; font-weight: 600; }
    p, span, div { color: #e0e0e0; font-family: 'Inter', 'Segoe UI', sans-serif; }
    
    /* Metric Cards: Hardware panel look */
    .stMetric > div { 
        background: #1e2129; 
        border: 1px solid #2d3038;
        border-radius: 6px; 
        padding: 16px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stMetric label { color: #a1aab8 !important; font-size: 0.9rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700 !important; }
    .stMetric [data-testid="stMetricDelta"] { color: #76b900 !important; }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #2d3038; }
    .stTabs [data-baseweb="tab"] { 
        background: transparent; 
        color: #a1aab8; 
        border: none;
        padding-bottom: 8px;
    }
    .stTabs [aria-selected="true"] { 
        color: #76b900 !important; 
        border-bottom: 2px solid #76b900 !important; 
        background: transparent !important;
    }
    
    /* Layout padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 95%; }
    
    /* Expanders */
    div[data-testid="stExpander"] { 
        background: #1e2129;
        border: 1px solid #2d3038; 
        border-radius: 6px; 
    }
    div[data-testid="stExpander"] summary { color: #76b900; }
</style>
""",
    unsafe_allow_html=True,
)


# ===========================================================
# DATA LOADING
# ===========================================================
@st.cache_data(ttl=30)
def load_test_data():
    for p in ["test_features.csv", "data/test_features.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


@st.cache_resource
def load_model():
    for p in ["models/isolation_forest.joblib", "isolation_forest_model.joblib"]:
        if os.path.exists(p):
            model = IsolationForestModel()
            model.load(p)
            return model
    return None


@st.cache_resource
def load_ledger():
    return BlackBoxLedger()


def load_benchmarks():
    for p in ["outputs/benchmarks/benchmark_results.json", "benchmark_results.json"]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


df_test = load_test_data()
model = load_model()
ledger = load_ledger()
benchmarks = load_benchmarks()

# Blockchain validation
is_secure, security_msg = ledger.validate_integrity()

# ===========================================================
# SIDEBAR
# ===========================================================
with st.sidebar:
    st.markdown("## ⚡ PulseNet")
    st.markdown("**Predictive Maintenance**")
    st.markdown("---")

    if df_test is not None:
        engine_ids = sorted(df_test["unit_number"].unique())
        selected_engine = st.selectbox("🔧 Select Engine Unit", engine_ids)
        st.caption(f"Monitoring Unit #{selected_engine}")
    else:
        st.error("⚠️ No data loaded. Run the pipeline first.")
        st.stop()

    st.markdown("---")

    # System status
    st.markdown("### System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", "✅" if model else "❌")
    with col2:
        st.metric("Chain", "🔒" if is_secure else "⚠️")

    st.caption(f"Ledger: {len(ledger.chain)} blocks")
    st.caption(f"Merkle: {ledger.compute_merkle_root()[:12]}...")

if df_test is None or model is None:
    st.error("Missing model or data. Please run the pipeline.")
    st.stop()

# ===========================================================
# DATA PROCESSING
# ===========================================================
engine_data = df_test[df_test["unit_number"] == selected_engine].copy()
feature_cols = [
    c
    for c in engine_data.columns
    if c not in ("unit_number", "time_in_cycles", "is_anomaly")
]
X_engine = engine_data[feature_cols]

try:
    health_scores = model.health_index(X_engine)
    engine_data["health_index"] = health_scores
except Exception:
    raw_scores = model.decision_function(X_engine)
    engine_data["health_index"] = np.clip(((raw_scores + 0.15) / 0.3) * 100, 0, 100)

engine_data["status"] = np.where(
    engine_data["health_index"] > 50, "Healthy", "Critical"
)

current_health = engine_data["health_index"].iloc[-1]
total_cycles = len(engine_data)
health_delta = (
    current_health - engine_data["health_index"].iloc[-2] if total_cycles > 1 else 0
)

# ===========================================================
# HEADER
# ===========================================================
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.title(f"Engine Unit #{selected_engine}")
    if is_secure:
        st.caption(
            f"🔒 BLOCKCHAIN SECURED  |  {len(ledger.chain)} blocks  |  ✅ VERIFIED"
        )
    else:
        st.error(f"🚨 INTEGRITY ALERT: {security_msg}")
        st.stop()

with col_badge:
    # NVIDIA Green = #76b900, Orange/Red for Attention
    badge_color = "#76b900" if current_health > 50 else "#e74c3c"
    badge_text = "OPTIMAL" if current_health > 50 else "ATTENTION"
    st.markdown(
        f"""
    <div style="text-align:right; padding:10px;">
        <span style="background:{badge_color}; color:#000000; padding:8px 16px;
                     border-radius:4px; font-weight:800; font-size:14px; text-transform:uppercase; letter-spacing:1px;">
            {badge_text}
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ===========================================================
# METRIC CARDS
# ===========================================================
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Health Score", f"{current_health:.1f}%", delta=f"{health_delta:.2f}%")
with m2:
    st.metric("Operating Cycles", f"{total_cycles}")
with m3:
    risk = (
        "Low" if current_health > 70 else ("Medium" if current_health > 50 else "High")
    )
    st.metric("Risk Level", risk, delta_color="inverse")
with m4:
    rul = max(0, 150 - total_cycles)
    st.metric("Est. RUL", f"{rul} cycles")

# ===========================================================
# TABBED CONTENT
# ===========================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Health & Trends",
        "🔍 Sensor Deep Dive",
        "⛓️ Blockchain Ledger",
        "📊 System Metrics",
    ]
)

# TAB 1: Health & Trends
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Health Degradation Curve")
        # NVIDIA Green = #76b900
        fill = "rgba(118,185,0,0.2)" if current_health > 50 else "rgba(231,76,60,0.2)"
        line = "#76b900" if current_health > 50 else "#e74c3c"
        fig = px.area(
            engine_data,
            x="time_in_cycles",
            y="health_index",
            template="plotly_dark",
            height=380,
        )
        fig.update_traces(line_color=line, fillcolor=fill)
        fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="#e74c3c",
            annotation_text="Critical Threshold",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Cycle",
            yaxis_title="Health Index (%)",
            xaxis={"showgrid": True, "gridcolor": "#2d3038"},
            yaxis={"showgrid": True, "gridcolor": "#2d3038"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Recent Telemetry")
        recent = engine_data[["time_in_cycles", "health_index", "status"]].tail(10)
        recent = recent.sort_values("time_in_cycles", ascending=False)
        st.dataframe(
            recent,
            hide_index=True,
            height=380,
            column_config={
                "health_index": st.column_config.ProgressColumn(
                    "Health", format="%.1f%%", min_value=0, max_value=100
                ),
            },
            use_container_width=True,
        )

    # Anomaly alerts
    critical = engine_data[engine_data["status"] == "Critical"]
    if len(critical) > 0:
        st.warning(
            f"⚠️ {len(critical)} critical cycles detected for Unit #{selected_engine}"
        )

# TAB 2: Sensor Deep Dive
with tab2:
    sensor_opts = [str(c) for c in feature_cols if "sensor" in str(c) and "rolling" not in str(c)]
    selected_sensors = st.multiselect(
        "Select sensors to compare:",
        sensor_opts,
        default=sensor_opts[:3] if sensor_opts else [],
    )
    if selected_sensors:
        fig_s = px.line(
            engine_data,
            x="time_in_cycles",
            y=selected_sensors,
            template="plotly_dark",
            height=400,
        )
        fig_s.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_s, use_container_width=True)

    # Multi-engine comparison
    st.subheader("Multi-Engine Health Comparison")
    all_healths = []
    for uid in sorted(df_test["unit_number"].unique())[:20]:
        udata = df_test[df_test["unit_number"] == uid]
        X_u = udata[feature_cols]
        try:
            h = model.health_index(X_u)
        except Exception:
            h = np.clip(((model.decision_function(X_u) + 0.15) / 0.3) * 100, 0, 100)
        
        h_list = list(h)  # type: ignore
        all_healths.append({"unit": int(uid), "health": float(h_list[-1])})

    health_df = pd.DataFrame(all_healths)
    fig_multi = px.bar(
        health_df,
        x="unit",
        y="health",
        template="plotly_dark",
        color="health",
        color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
        height=300,
    )
    fig_multi.add_hline(y=50, line_dash="dot", line_color="#e74c3c")
    fig_multi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_multi, use_container_width=True)

# TAB 3: Blockchain
with tab3:
    st.subheader("⛓️ Blockchain Integrity")
    if is_secure:
        st.success(
            f"✅ All {len(ledger.chain)} blocks verified. Chain is cryptographically valid."
        )
        st.metric("Merkle Root", ledger.compute_merkle_root()[:24] + "...")

        st.markdown("**Recent Blocks:**")
        for b in reversed(ledger.chain[-5:]):
            with st.expander(f"Block #{b.index}  |  Hash: {b.hash[:20]}..."):
                st.json(b.to_dict())
    else:
        st.error(f"🚨 Chain validation failed: {security_msg}")

# TAB 4: System Metrics
with tab4:
    if benchmarks:
        st.subheader("Performance Benchmarks")
        c1, c2 = st.columns(2)

        with c1:
            if "inference_latency" in benchmarks:
                lat = benchmarks["inference_latency"]
                st.markdown("### Inference Latency")
                st.metric("Median", f"{lat.get('median_ms', 'N/A')} ms")
                st.metric("P95", f"{lat.get('p95_ms', 'N/A')} ms")
                st.metric(
                    "Target (<50ms)", "✅ Met" if lat.get("target_met") else "❌ Missed"
                )

        with c2:
            if "throughput" in benchmarks:
                st.markdown("### Throughput")
                tp_df = pd.DataFrame(
                    [
                        {"Batch Size": k.replace("batch_", ""), "Samples/sec": v}
                        for k, v in benchmarks["throughput"].items()
                    ]
                )
                st.dataframe(tp_df, hide_index=True, use_container_width=True)

        if "network_resilience" in benchmarks:
            st.markdown("### Network Resilience")
            for k, v in benchmarks["network_resilience"].items():
                rate = k.replace("loss_", "").replace("pct", "%")
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.progress(v["data_integrity_pct"] / 100)
                with col_b:
                    st.write(f"{rate} loss → {v['data_integrity_pct']}% integrity")
    else:
        st.info(
            "No benchmark data found. Run `python main_pipeline.py --mode benchmark`"
        )

# ===========================================================
# FOOTER
# ===========================================================
st.markdown("---")
st.caption(
    "PulseNet v2.1 — Production Predictive Maintenance Platform  |  © 2026 Pooja Kiran, Rhutvik Pachghare"
)
