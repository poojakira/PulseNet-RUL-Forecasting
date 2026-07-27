"""
Enhanced Streamlit Dashboard — real-time anomaly monitoring, sensor trends,
blockchain status, system metrics, and multi-engine support.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from pulsenet.config import cfg  # noqa: E402
from pulsenet.logger import get_logger  # noqa: E402
from pulsenet.models.registry import ModelRegistry  # noqa: E402
from pulsenet.pipeline.preprocessing import (  # noqa: E402
    create_sequences,
)
from pulsenet.security.blockchain import BlackBoxLedger  # noqa: E402

log = get_logger(__name__)

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
    h1, h2, h3 {
        color: #76b900 !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    p, span, div {
        color: #e0e0e0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Metric Cards: Hardware panel look */
    .stMetric > div {
        background: #1e2129;
        border: 1px solid #2d3038;
        border-radius: 6px;
        padding: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stMetric label { color: #a1aab8 !important; font-size: 0.9rem !important; }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
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

    .ops-note {
        color: #a1aab8;
        font-size: 0.86rem;
        line-height: 1.35;
        margin-top: -0.5rem;
        margin-bottom: 0.75rem;
    }

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
# OPERATIONAL SCORING HELPERS
# ===========================================================
def score_unit_health(
    unit_data: pd.DataFrame, feature_columns: list[str], model_obj: Any
) -> np.ndarray:
    """Score one asset and return health aligned to its cycle rows."""
    if unit_data.empty:
        return np.array([100.0])

    active_model_name = cfg.models.active_model
    try:
        if active_model_name in ("lstm", "transformer"):
            seq_len = int(cfg.models.lstm.sequence_length)
            X_seq = create_sequences(
                cast(pd.DataFrame, unit_data), feature_columns, seq_len=seq_len
            )
            if len(X_seq) == 0:
                return np.full(len(unit_data), 100.0)
            health_scores = np.asarray(model_obj.health_index(X_seq), dtype=float)
            padded = np.concatenate([np.full(seq_len - 1, 100.0), health_scores])
            return padded[: len(unit_data)]

        X_unit = unit_data[feature_columns]
        return np.asarray(model_obj.health_index(np.asarray(X_unit)), dtype=float)
    except Exception as e:
        log.warning(f"Health scoring failed for unit telemetry: {e}")
        return np.full(len(unit_data), 100.0)


def estimate_rul(health_values: list[float]) -> Any:
    """Estimate remaining useful life from recent health slope."""
    current = float(health_values[-1]) if health_values else 0.0
    if len(health_values) <= 10 or current >= 95:
        return "> 100"

    recent_health = health_values[-10:]
    recent_cycles = np.arange(len(recent_health))
    poly_res = np.polyfit(recent_cycles, recent_health, 1)
    slope = float(poly_res[0])
    if slope < 0:
        return int(max(0, -current / slope))
    return "Stable"


def sensor_drift_score(unit_data: pd.DataFrame, sensor_columns: list[str]) -> float:
    """Return 0-100 mean z-score drift between baseline and recent telemetry."""
    numeric_sensors = [
        c
        for c in sensor_columns
        if c in unit_data and pd.api.types.is_numeric_dtype(unit_data[c])
    ]
    if len(unit_data) < 8 or not numeric_sensors:
        return 0.0

    baseline_end = max(4, int(len(unit_data) * 0.6))
    recent_window = max(4, min(15, len(unit_data) // 4))
    baseline = unit_data.iloc[:baseline_end][numeric_sensors].astype(float)
    recent = unit_data.tail(recent_window)[numeric_sensors].astype(float)
    std = baseline.std().replace(0, np.nan)
    drift = ((recent.mean() - baseline.mean()).abs() / std).replace(
        [np.inf, -np.inf], np.nan
    )
    return float(np.clip(drift.fillna(0.0).mean() * 20.0, 0.0, 100.0))


def anomaly_hits(unit_data: pd.DataFrame, health_values: list[float]) -> int:
    """Count recent anomaly flags, falling back to critical health cycles."""
    if "is_anomaly" in unit_data:
        recent_flags = pd.to_numeric(
            unit_data["is_anomaly"].tail(20), errors="coerce"
        )
        return int(recent_flags.sum())
    return int(sum(1 for h in health_values[-20:] if h < 50.0))


def maintenance_priority(
    health: float, drift: float, rul: Any, anomalies: int, integrity_ok: bool
) -> int:
    """Combine health, drift, RUL pressure, anomaly count, and chain state."""
    if isinstance(rul, int):
        rul_pressure = float(np.clip((90 - rul) * 0.55, 0.0, 50.0))
    else:
        rul_pressure = 0.0
    integrity_pressure = 25.0 if not integrity_ok else 0.0
    raw_score = (
        (100.0 - health) * 0.42
        + drift * 0.32
        + min(anomalies, 6) * 5.5
        + rul_pressure
        + integrity_pressure
    )
    return int(np.clip(round(raw_score), 0, 100))


def operational_status(
    health: float, drift: float, rul: Any, anomalies: int, integrity_ok: bool
) -> str:
    if not integrity_ok:
        return "Integrity Review"
    if health < 50 or (isinstance(rul, int) and rul <= 20):
        return "Immediate"
    if anomalies > 0 or drift >= 55:
        return "Investigate"
    if health < 70 or (isinstance(rul, int) and rul <= 60):
        return "Schedule"
    return "Normal"


@st.cache_data(ttl=30, show_spinner=False)
def build_fleet_snapshot(
    raw_data: pd.DataFrame,
    feature_columns: list[str],
    sensor_columns: list[str],
    fleet_cap: int,
    selected_unit: int,
    integrity_ok: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    units = sorted(int(x) for x in raw_data["unit_number"].unique())
    included_units = units[:fleet_cap]
    if selected_unit in units and selected_unit not in included_units:
        included_units = [*included_units[:-1], selected_unit]
    for uid in included_units:
        unit_data = raw_data[raw_data["unit_number"] == uid].copy()
        health = score_unit_health(unit_data, feature_columns, model).tolist()
        current = float(health[-1]) if health else 0.0
        delta = float(current - health[-2]) if len(health) > 1 else 0.0
        rul = estimate_rul([float(x) for x in health])
        drift = sensor_drift_score(unit_data, sensor_columns)
        anomalies = anomaly_hits(unit_data, [float(x) for x in health])
        priority = maintenance_priority(current, drift, rul, anomalies, integrity_ok)
        status = operational_status(current, drift, rul, anomalies, integrity_ok)
        latest_cycle = int(unit_data["time_in_cycles"].max())
        rows.append(
            {
                "unit": int(uid),
                "cycle": latest_cycle,
                "health": current,
                "health_delta": delta,
                "rul": rul,
                "rul_plot": float(rul if isinstance(rul, int) else 120),
                "drift": drift,
                "anomalies": anomalies,
                "priority": priority,
                "status": status,
                "security_overlay": (not integrity_ok) or anomalies > 0 or drift >= 55,
            }
        )
    return pd.DataFrame(rows)


def render_fleet_ops_3d(fleet_data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    status_colors = {
        "Normal": "#76b900",
        "Schedule": "#f4c542",
        "Investigate": "#ff8a3d",
        "Immediate": "#e74c3c",
        "Integrity Review": "#ff2d55",
    }

    for status, group in fleet_data.groupby("status", sort=False):
        fig.add_trace(
            go.Scatter3d(
                x=group["cycle"],
                y=group["drift"],
                z=group["rul_plot"],
                mode="markers",
                name=status,
                marker={
                    "size": np.clip(group["priority"] / 6 + 5, 6, 18),
                    "color": status_colors.get(str(status), "#8a93a5"),
                    "opacity": 0.9,
                    "line": {"color": "#0f1115", "width": 1},
                },
                customdata=np.stack(
                    [
                        group["unit"],
                        group["health"].round(1),
                        group["priority"],
                        group["anomalies"],
                        group["status"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Unit %{customdata[0]}<br>"
                    "Health %{customdata[1]}%<br>"
                    "Priority %{customdata[2]}/100<br>"
                    "Anomalies %{customdata[3]}<br>"
                    "State %{customdata[4]}<extra></extra>"
                ),
            )
        )

    overlay = fleet_data[fleet_data["security_overlay"]]
    if not overlay.empty:
        fig.add_trace(
            go.Scatter3d(
                x=overlay["cycle"],
                y=overlay["drift"],
                z=overlay["rul_plot"],
                mode="markers",
                name="Anomaly / Security Overlay",
                marker={
                    "size": np.clip(overlay["priority"] / 5 + 9, 10, 24),
                    "color": "#ff2d55",
                    "symbol": "diamond",
                    "opacity": 0.55,
                    "line": {"color": "#ffffff", "width": 1},
                },
                hoverinfo="skip",
            )
        )

    if not fleet_data.empty:
        x_min, x_max = fleet_data["cycle"].min(), fleet_data["cycle"].max()
        y_min, y_max = 0, max(75.0, float(fleet_data["drift"].max()) + 10.0)
        fig.add_trace(
            go.Surface(
                x=[[x_min, x_max], [x_min, x_max]],
                y=[[y_min, y_min], [y_max, y_max]],
                z=[[30, 30], [30, 30]],
                name="30-cycle service horizon",
                opacity=0.18,
                showscale=False,
                colorscale=[[0, "#e74c3c"], [1, "#e74c3c"]],
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.02, "x": 0},
        scene={
            "xaxis_title": "Current Cycle",
            "yaxis_title": "Sensor Drift",
            "zaxis_title": "RUL Horizon",
            "bgcolor": "rgba(0,0,0,0)",
            "xaxis": {"gridcolor": "#2d3038"},
            "yaxis": {"gridcolor": "#2d3038"},
            "zaxis": {"gridcolor": "#2d3038", "range": [0, 125]},
            "camera": {"eye": {"x": 1.55, "y": 1.35, "z": 0.95}},
        },
    )
    return fig

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
    active_name = cfg.models.active_model
    registry = ModelRegistry()

    try:
        model = registry.get_model(active_name)
        model_paths = [
            Path(f"models/{active_name}.joblib"),
            Path(f"data/models/{active_name}.joblib"),
            Path(f"{active_name}_model.joblib"),
        ]

        for p in model_paths:
            if p.exists():
                model.load(p)
                return model
        return None
    except Exception as e:
        st.error(f"Failed to load model '{active_name}': {e}")
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

# Initial state for typing safety
selected_engine: int = 1
fleet_limit: int = 40
is_secure: bool = False
security_msg: str = ""

if ledger:
    # We'll set tenant_id from sidebar below, so this moves.
    pass

# ===========================================================
# SIDEBAR
# ===========================================================
with st.sidebar:
    st.markdown("## ⚡ PulseNet")
    st.markdown("**Predictive Maintenance**")
    st.markdown("---")

    if df_test is not None:
        unit_nums = df_test["unit_number"].unique()
        engine_ids = sorted([int(x) for x in unit_nums])
        selected_engine = int(st.selectbox("🔧 Select Engine Unit", engine_ids) or 1)
        fleet_limit = int(
            st.slider(
                "3D Fleet Window",
                min_value=1,
                max_value=min(100, len(engine_ids)),
                value=min(40, len(engine_ids)),
            )
        )
        st.caption(f"Monitoring Unit #{selected_engine}")
    else:
        st.error("⚠️ No data loaded. Run the pipeline first.")
    st.markdown("---")
    st.markdown("### 🏢 Multitenancy")
    tenant_id = st.text_input("Enter Tenant ID", value="public").strip().lower()
    st.caption(f"Context: {tenant_id.upper()}")

    if ledger:
        is_secure, security_msg = ledger.validate_integrity(tenant_id)

    st.markdown("---")

    # System status
    st.markdown("### System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", "✅" if model else "❌")
    with col2:
        st.metric("Chain", "🔒" if is_secure else "⚠️")

    metrics = ledger.get_metrics(tenant_id) if ledger else {}
    st.caption(f"Ledger: {metrics.get('total_blocks_global', 0)} total blocks")
    st.caption(f"Merkle: {ledger.compute_merkle_root(tenant_id)[:12]}...")

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
sensor_opts = [
    str(c) for c in feature_cols if "sensor" in str(c) and "rolling" not in str(c)
]

health_scores = score_unit_health(engine_data, feature_cols, model)
engine_data["health_index"] = health_scores
engine_data["status"] = np.where(
    engine_data["health_index"] > 50, "Healthy", "Critical"
)

health_series = [float(x) for x in engine_data["health_index"].tolist()]
current_health = float(health_series[-1]) if health_series else 0.0
total_cycles = len(engine_data)
health_delta = float(current_health - health_series[-2]) if total_cycles > 1 else 0.0
est_rul = estimate_rul(health_series)

fleet_snapshot = build_fleet_snapshot(
    df_test, feature_cols, sensor_opts, fleet_limit, selected_engine, is_secure
)
if not fleet_snapshot.empty and selected_engine in set(fleet_snapshot["unit"]):
    current_asset = fleet_snapshot[fleet_snapshot["unit"] == selected_engine].iloc[0]
    current_priority = int(current_asset["priority"])
    current_drift = float(current_asset["drift"])
    current_status = str(current_asset["status"])
else:
    current_priority = maintenance_priority(current_health, 0.0, est_rul, 0, is_secure)
    current_drift = 0.0
    current_status = operational_status(current_health, 0.0, est_rul, 0, is_secure)

fleet_urgent = int(
    fleet_snapshot["status"].isin(["Immediate", "Integrity Review"]).sum()
)
security_overlays = int(fleet_snapshot["security_overlay"].sum())

# ===========================================================
# HEADER
# ===========================================================
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.title(f"Engine Unit #{selected_engine}")
    if is_secure:
        metrics = ledger.get_metrics(tenant_id)
        st.caption(
            f"🔒 BLOCKCHAIN SECURED  |  Tenant: {tenant_id.upper()}  |  ✅ VERIFIED"
        )
    else:
        st.error(f"🚨 INTEGRITY ALERT: {security_msg}")
        st.warning("Operational view is read-only until ledger integrity recovers.")

with col_badge:
    # NVIDIA Green = #76b900, Orange/Red for Attention
    badge_color = "#76b900" if current_health > 50 else "#e74c3c"
    badge_text = "OPTIMAL" if current_health > 50 else "ATTENTION"
    st.markdown(
        f"""
    <div style="text-align:right; padding:10px;">
        <span style="
            background:{badge_color};
            color:#000000;
            padding:8px 16px;
            border-radius:4px;
            font-weight:800;
            font-size:14px;
            text-transform:uppercase;
            letter-spacing:1px;">
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
m1, m2, m3, m4, m5 = st.columns(5)
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
    st.metric("Est. RUL", f"{est_rul} cycles" if isinstance(est_rul, int) else est_rul)
with m5:
    st.metric("Ops Priority", f"{current_priority}/100", delta=current_status)

# ===========================================================
# TABBED CONTENT
# ===========================================================
tab_ops, tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🛰️ 3D Ops View",
        "📈 Health & Trends",
        "🔍 Sensor Deep Dive",
        "⛓️ Blockchain Ledger",
        "📊 System Metrics",
    ]
)


# TAB 0: 3D Operational View
with tab_ops:
    st.subheader("Fleet Operations Map")
    st.markdown(
        """
        <div class="ops-note">
        X is current operating cycle, Y is sensor drift, Z is RUL horizon.
        Marker size is maintenance priority; red diamond overlays mark anomaly,
        drift, or integrity-review signals.
        </div>
        """,
        unsafe_allow_html=True,
    )

    ops_a, ops_b, ops_c, ops_d = st.columns(4)
    with ops_a:
        st.metric("Assets in View", len(fleet_snapshot))
    with ops_b:
        st.metric("Immediate Queue", fleet_urgent)
    with ops_c:
        st.metric("Security Overlays", security_overlays)
    with ops_d:
        st.metric("Selected Drift", f"{current_drift:.1f}/100")

    st.plotly_chart(render_fleet_ops_3d(fleet_snapshot), use_container_width=True)

    q1, q2 = st.columns([1.15, 1])
    with q1:
        st.subheader("Maintenance Priority Queue")
        queue_cols = [
            "unit",
            "status",
            "priority",
            "health",
            "rul",
            "drift",
            "anomalies",
        ]
        queue_df = fleet_snapshot.sort_values(
            by=["priority", "drift"], ascending=[False, False]
        )[queue_cols].head(12)
        st.dataframe(
            queue_df,
            hide_index=True,
            height=340,
            use_container_width=True,
            column_config={
                "health": st.column_config.ProgressColumn(
                    "Health", format="%.1f%%", min_value=0, max_value=100
                ),
                "drift": st.column_config.ProgressColumn(
                    "Drift", format="%.1f", min_value=0, max_value=100
                ),
                "priority": st.column_config.ProgressColumn(
                    "Priority", format="%d", min_value=0, max_value=100
                ),
            },
        )

    with q2:
        st.subheader("RUL Distribution")
        fig_rul = px.histogram(
            fleet_snapshot,
            x="rul_plot",
            color="status",
            nbins=18,
            template="plotly_dark",
            height=340,
            color_discrete_map={
                "Normal": "#76b900",
                "Schedule": "#f4c542",
                "Investigate": "#ff8a3d",
                "Immediate": "#e74c3c",
                "Integrity Review": "#ff2d55",
            },
        )
        fig_rul.add_vline(x=30, line_dash="dot", line_color="#e74c3c")
        fig_rul.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="RUL horizon; stable assets bucket at 120",
            yaxis_title="Assets",
            legend_title_text="",
        )
        st.plotly_chart(fig_rul, use_container_width=True)

    drift_view = fleet_snapshot.sort_values("drift", ascending=False).head(15)
    fig_drift = px.bar(
        drift_view,
        x="unit",
        y="drift",
        color="priority",
        template="plotly_dark",
        color_continuous_scale=["#76b900", "#f4c542", "#e74c3c"],
        height=280,
        labels={"unit": "Unit", "drift": "Sensor Drift", "priority": "Priority"},
    )
    fig_drift.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"type": "category", "gridcolor": "#2d3038"},
        yaxis={"gridcolor": "#2d3038", "range": [0, 100]},
    )
    st.plotly_chart(fig_drift, use_container_width=True)
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
        selected_cols = ["time_in_cycles", "health_index", "status"]
        recent = cast(pd.DataFrame, engine_data[selected_cols]).tail(10)
        recent = recent.sort_values(by="time_in_cycles", ascending=False)
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
    fig_multi = px.bar(
        fleet_snapshot,
        x="unit",
        y="health",
        template="plotly_dark",
        color="priority",
        color_continuous_scale=["#76b900", "#f4c542", "#e74c3c"],
        height=300,
        labels={"unit": "Unit", "health": "Health", "priority": "Priority"},
    )
    fig_multi.add_hline(y=50, line_dash="dot", line_color="#e74c3c")
    fig_multi.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"type": "category", "gridcolor": "#2d3038"},
        yaxis={"gridcolor": "#2d3038", "range": [0, 100]},
    )
    st.plotly_chart(fig_multi, use_container_width=True)

# TAB 3: Blockchain
with tab3:
    st.subheader("⛓️ Blockchain Integrity")
    if is_secure:
        recent_blocks = ledger.get_recent_blocks(10, tenant_id)
        st.success(
            "✅ Ledger verified for "
            f"{tenant_id.upper()}. Chain is cryptographically valid."
        )
        st.metric("Merkle Root", ledger.compute_merkle_root(tenant_id)[:24] + "...")

        st.markdown("**Recent Blocks:**")
        for b in reversed(recent_blocks):
            # Block is dict from get_recent_blocks
            with st.expander(f"Block #{b['index']}  |  Hash: {b['hash'][:20]}..."):
                st.json(b)
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

        st.markdown("---")
        st.subheader("🎯 Detection Quality & Reliability")
        q1, q2, q3 = st.columns(3)

        if "detection_quality" in benchmarks:
            dq = benchmarks["detection_quality"]
            with q1:
                st.metric("Anomaly F1-Score", f"{dq.get('f1', 0.0):.3f}")
            with q2:
                st.metric("Recall (Sensitivity)", f"{dq.get('recall', 0.0):.1%}")

        if "lead_time" in benchmarks:
            lt = benchmarks["lead_time"]
            with q3:
                st.metric("Avg Lead Time", f"{lt.get('avg_lead_time', 0.0)} cycles")

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
    "PulseNet v2.1 — Production Predictive Maintenance Platform"
    "  |  \u00a9 2026 Pooja Kiran, Rhutvik Pachghare"
)
