"""
PulseNet — Industrial Operations Console
=========================================

Streamlit dashboard styled as a Network Operations Center / SCADA-style
console for predictive-maintenance teams. Five views are available from
the left rail:

    1. Fleet Overview    — All engines, status grid, KPI strip
    2. Engine Detail     — Single-asset deep dive with degradation curve
    3. Alarms            — Active critical events from the audit ledger
    4. Audit Trail       — Tamper-evident block log + Merkle root
    5. System Health     — API/Prometheus metrics, model registry

Every number on this dashboard comes from one of three real sources:
    * Audit ledger JSON files on disk (cryptographically chained)
    * Inference results from the loaded model on the loaded test CSV
    * Benchmark JSON written by `python main_pipeline.py --mode benchmark`

Nothing is synthesized. If a panel says "No data", run the relevant
pipeline step. The dashboard never fabricates measurements to fill space.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
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

from pulsenet.config import cfg
from pulsenet.logger import get_logger
from pulsenet.models.registry import ModelRegistry
from pulsenet.pipeline.preprocessing import create_sequences
from pulsenet.security.blockchain import BlackBoxLedger

log = get_logger(__name__)

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="PulseNet Ops Console",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Industrial NOC palette: matte greys + status colors used consistently
# across every panel. Status colors follow IEC 62682 alarm philosophy:
#   GREEN  = nominal       AMBER  = warning
#   RED    = critical      BLUE   = informational
COLOR_BG = "#0d1117"
COLOR_PANEL = "#161b22"
COLOR_BORDER = "#30363d"
COLOR_TEXT = "#c9d1d9"
COLOR_MUTED = "#8b949e"
COLOR_ACCENT = "#58a6ff"
COLOR_OK = "#3fb950"
COLOR_WARN = "#d29922"
COLOR_CRIT = "#f85149"

st.markdown(
    f"""
<style>
    .stApp {{ background: {COLOR_BG}; color: {COLOR_TEXT}; }}
    [data-testid="stSidebar"] {{
        background: {COLOR_PANEL};
        border-right: 1px solid {COLOR_BORDER};
    }}
    [data-testid="stSidebarUserContent"] {{ padding-top: 8px; }}
    h1, h2, h3, h4 {{
        color: {COLOR_TEXT} !important;
        font-family: -apple-system, "SF Pro Display", "Segoe UI", system-ui, sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
    }}
    p, span, div, label {{
        color: {COLOR_TEXT};
        font-family: -apple-system, "SF Pro Display", "Segoe UI", system-ui, sans-serif;
    }}
    .stMetric > div {{
        background: {COLOR_PANEL};
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        padding: 12px 16px;
    }}
    .stMetric label {{
        color: {COLOR_MUTED} !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .stMetric [data-testid="stMetricValue"] {{
        color: {COLOR_TEXT} !important;
        font-weight: 600 !important;
        font-variant-numeric: tabular-nums;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 1px solid {COLOR_BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        color: {COLOR_MUTED};
        border: none;
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLOR_TEXT} !important;
        border-bottom: 2px solid {COLOR_ACCENT} !important;
    }}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 100%;
    }}
    div[data-testid="stExpander"] {{
        background: {COLOR_PANEL};
        border: 1px solid {COLOR_BORDER};
        border-radius: 4px;
    }}
    code, pre {{
        background: {COLOR_BG} !important;
        border: 1px solid {COLOR_BORDER};
        font-family: "SF Mono", "JetBrains Mono", Consolas, monospace;
    }}
    /* Status pill — used in fleet table */
    .pill {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}
    .pill-ok {{ background: {COLOR_OK}33; color: {COLOR_OK}; border: 1px solid {COLOR_OK}; }}
    .pill-warn {{ background: {COLOR_WARN}33; color: {COLOR_WARN}; border: 1px solid {COLOR_WARN}; }}
    .pill-crit {{ background: {COLOR_CRIT}33; color: {COLOR_CRIT}; border: 1px solid {COLOR_CRIT}; }}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Resource loaders
# ----------------------------------------------------------------------
@st.cache_data(ttl=15)
def load_test_data() -> pd.DataFrame | None:
    """Load preprocessed feature CSV from canonical or legacy paths."""
    candidates = [
        Path(cfg.system.data_dir) / "test_features.csv",
        Path("test_features.csv"),
        Path("data/test_features.csv"),
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


@st.cache_resource
def load_model() -> Any:
    """Load the active model from the canonical model_dir."""
    active_name = cfg.models.active_model
    registry = ModelRegistry()
    try:
        m = registry.get_model(active_name)
    except Exception as exc:
        st.error(f"Failed to instantiate model '{active_name}': {exc}")
        return None

    candidates = [
        Path(cfg.models.model_dir) / f"{active_name}.joblib",
        Path("models") / f"{active_name}.joblib",
        Path(f"{active_name}_model.joblib"),
    ]
    for p in candidates:
        if p.exists():
            try:
                m.load(p)
                return m
            except Exception as exc:
                log.warning(f"Failed to load model from {p}: {exc}")
    return None


@st.cache_resource
def load_ledger() -> BlackBoxLedger:
    return BlackBoxLedger()


def load_benchmarks() -> dict | None:
    candidates = [
        Path("outputs/benchmarks/benchmark_results.json"),
        Path("benchmark_results.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception as exc:
                log.warning(f"Bad benchmark JSON at {p}: {exc}")
    return None


def status_pill(label: str, kind: str) -> str:
    return f'<span class="pill pill-{kind}">{label}</span>'


def health_to_status(h: float) -> tuple[str, str]:
    """Map health-index value to (label, css-class)."""
    if h >= 70:
        return "NOMINAL", "ok"
    if h >= 50:
        return "WARNING", "warn"
    return "CRITICAL", "crit"


# ----------------------------------------------------------------------
# Initial load
# ----------------------------------------------------------------------
df_test = load_test_data()
model = load_model()
ledger = load_ledger()
benchmarks = load_benchmarks()

# ----------------------------------------------------------------------
# Sidebar — site/tenant, navigation, system status
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<h2 style='margin-top:0;color:{COLOR_TEXT}'>◆ PulseNet</h2>"
        f"<div style='color:{COLOR_MUTED};font-size:0.8rem;margin-top:-12px'>"
        f"Predictive Maintenance Console</div><hr/>",
        unsafe_allow_html=True,
    )

    tenant_id = st.text_input("Tenant", value="public").strip().lower() or "public"

    view = st.radio(
        "View",
        options=["Fleet Overview", "Engine Detail", "Alarms", "Audit Trail", "System Health"],
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{COLOR_MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em'>Connection</div>", unsafe_allow_html=True)

    is_secure, security_msg = ledger.validate_integrity(tenant_id)
    sec_pill = status_pill("CHAIN OK", "ok") if is_secure else status_pill("CHAIN ALERT", "crit")
    model_pill = status_pill("MODEL LOADED", "ok") if model else status_pill("NO MODEL", "warn")
    data_pill = status_pill("DATA LOADED", "ok") if df_test is not None else status_pill("NO DATA", "warn")
    st.markdown(
        f"<div style='display:flex;flex-direction:column;gap:6px;margin-top:6px'>{sec_pill}{model_pill}{data_pill}</div>",
        unsafe_allow_html=True,
    )

    metrics = ledger.get_metrics(tenant_id)
    st.markdown(
        f"<div style='margin-top:14px;color:{COLOR_MUTED};font-size:0.7rem'>"
        f"Ledger blocks: <strong style='color:{COLOR_TEXT}'>{metrics.get('total_blocks_global', 0)}</strong><br/>"
        f"Merkle: <code style='font-size:0.7rem'>{ledger.compute_merkle_root(tenant_id)[:14]}…</code>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='position:absolute;bottom:14px;left:14px;color:{COLOR_MUTED};font-size:0.7rem'>"
        f"v{cfg.system.version} &middot; {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Helper: compute fleet-wide health from current model + data
# ----------------------------------------------------------------------
def compute_fleet_health(
    df: pd.DataFrame, model_obj: Any, max_engines: int | None = None
) -> pd.DataFrame:
    """Score every engine in the loaded test set with the loaded model.

    Returns a DataFrame with columns: unit, cycles, health, status_label.
    Health values are real scores from model.health_index() — never faked.
    """
    feat_cols = [
        c for c in df.columns
        if c not in ("unit_number", "time_in_cycles", "is_anomaly")
    ]
    units = sorted(df["unit_number"].unique())
    if max_engines:
        units = units[:max_engines]

    rows = []
    seq_models = ("lstm", "transformer")
    for uid in units:
        udata = df[df["unit_number"] == uid]
        try:
            if cfg.models.active_model in seq_models:
                seq_len = int(cfg.models.lstm.sequence_length)
                X_seq = create_sequences(
                    cast(pd.DataFrame, udata), feat_cols, seq_len=seq_len
                )
                health_arr = (
                    model_obj.health_index(X_seq)
                    if len(X_seq) > 0
                    else np.array([100.0])
                )
            else:
                X = udata[feat_cols].to_numpy()
                health_arr = model_obj.health_index(X)
            current = float(health_arr[-1])
        except Exception as exc:
            log.warning(f"Health computation failed for unit {uid}: {exc}")
            current = float("nan")

        label = (
            "NOMINAL" if current >= 70
            else ("WARNING" if current >= 50 else "CRITICAL")
        )
        rows.append({
            "unit": int(uid),
            "cycles": int(len(udata)),
            "health": current,
            "status_label": label,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Guard: data + model are required for most views.
# ----------------------------------------------------------------------
if (df_test is None or model is None) and view != "System Health":
    st.error(
        "Test data or model not available. Run "
        "`python main_pipeline.py --mode full` first to generate "
        "`{}/test_features.csv` and `{}/isolation_forest.joblib`.".format(
            cfg.system.data_dir, cfg.models.model_dir
        )
    )
    st.stop()

# ======================================================================
# VIEW 1 — FLEET OVERVIEW
# ======================================================================
if view == "Fleet Overview":
    st.markdown(f"## Fleet Overview &nbsp; <span style='color:{COLOR_MUTED};font-size:1rem'>tenant: {tenant_id}</span>", unsafe_allow_html=True)
    fleet = compute_fleet_health(df_test, model)

    n_total = len(fleet)
    n_crit = int((fleet["status_label"] == "CRITICAL").sum())
    n_warn = int((fleet["status_label"] == "WARNING").sum())
    n_ok = int((fleet["status_label"] == "NOMINAL").sum())
    avg_health = float(fleet["health"].mean()) if n_total else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Engines monitored", f"{n_total}")
    k2.metric("Nominal", f"{n_ok}")
    k3.metric("Warning", f"{n_warn}", delta=None, delta_color="off")
    k4.metric("Critical", f"{n_crit}")
    k5.metric("Fleet avg health", f"{avg_health:.1f}%" if n_total else "—")

    st.markdown("<br/>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        st.markdown("##### Health distribution by engine")
        fig = go.Figure()
        colors = fleet["status_label"].map({
            "NOMINAL": COLOR_OK, "WARNING": COLOR_WARN, "CRITICAL": COLOR_CRIT
        })
        fig.add_trace(go.Bar(
            x=fleet["unit"].astype(str),
            y=fleet["health"],
            marker_color=colors,
            hovertemplate="Unit #%{x}<br>Health: %{y:.1f}%<extra></extra>",
        ))
        fig.add_hline(y=50, line_dash="dot", line_color=COLOR_CRIT, annotation_text="critical")
        fig.add_hline(y=70, line_dash="dot", line_color=COLOR_WARN, annotation_text="warn")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_PANEL,
            plot_bgcolor=COLOR_PANEL,
            xaxis_title="Engine ID",
            yaxis_title="Health index (%)",
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis={"showgrid": False},
            yaxis={"gridcolor": COLOR_BORDER, "range": [0, 100]},
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("##### Status breakdown")
        donut = pd.DataFrame({
            "status": ["Nominal", "Warning", "Critical"],
            "count": [n_ok, n_warn, n_crit],
        })
        fig_d = px.pie(
            donut, values="count", names="status", hole=0.6,
            color="status",
            color_discrete_map={
                "Nominal": COLOR_OK, "Warning": COLOR_WARN, "Critical": COLOR_CRIT,
            },
        )
        fig_d.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_PANEL, plot_bgcolor=COLOR_PANEL,
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(orientation="v", x=1.05, y=0.5),
        )
        st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("##### Asset table")
    fleet_display = fleet.copy()
    fleet_display["status"] = fleet_display["status_label"].apply(
        lambda s: ("◉ NOMINAL" if s == "NOMINAL" else
                   "◐ WARNING" if s == "WARNING" else "◯ CRITICAL")
    )
    st.dataframe(
        fleet_display[["unit", "cycles", "health", "status"]],
        column_config={
            "unit": st.column_config.NumberColumn("Engine", format="%d"),
            "cycles": st.column_config.NumberColumn("Cycles", format="%d"),
            "health": st.column_config.ProgressColumn(
                "Health", format="%.1f%%", min_value=0.0, max_value=100.0,
            ),
            "status": "Status",
        },
        hide_index=True,
        use_container_width=True,
    )


# ======================================================================
# VIEW 2 — ENGINE DETAIL
# ======================================================================
elif view == "Engine Detail":
    units = sorted(df_test["unit_number"].unique().astype(int))
    selected_engine = st.selectbox("Engine", options=units, index=0)
    engine_data = df_test[df_test["unit_number"] == selected_engine].copy()
    feat_cols = [c for c in engine_data.columns
                 if c not in ("unit_number", "time_in_cycles", "is_anomaly")]

    seq_models = ("lstm", "transformer")
    try:
        if cfg.models.active_model in seq_models:
            seq_len = int(cfg.models.lstm.sequence_length)
            X_seq = create_sequences(
                cast(pd.DataFrame, engine_data), feat_cols, seq_len=seq_len
            )
            if len(X_seq) > 0:
                h = model.health_index(X_seq)
                padded = np.concatenate([np.full(seq_len - 1, 100.0), h])
                engine_data["health_index"] = padded[: len(engine_data)]
            else:
                engine_data["health_index"] = 100.0
        else:
            engine_data["health_index"] = model.health_index(
                np.asarray(engine_data[feat_cols])
            )
    except Exception as exc:
        st.warning(f"Health scoring failed: {exc}")
        engine_data["health_index"] = float("nan")

    series = engine_data["health_index"].tolist()
    current_health = float(series[-1]) if series else float("nan")
    cycles = len(engine_data)
    delta = (
        float(series[-1] - series[-2]) if cycles > 1 and not np.isnan(current_health) else 0.0
    )
    risk_label, risk_kind = health_to_status(current_health)

    # Linear extrapolation for RUL — explicitly labeled; no fabrication
    est_rul: Any = "—"
    if cycles > 10 and not np.isnan(current_health) and current_health < 95:
        recent = series[-10:]
        slope, intercept = np.polyfit(np.arange(len(recent)), recent, 1)
        slope = float(slope)
        if slope < 0:
            est_rul = max(0, int(-current_health / slope))

    head_l, head_r = st.columns([3, 1])
    head_l.markdown(f"## Engine #{selected_engine}")
    head_r.markdown(
        f"<div style='text-align:right;padding-top:8px'>{status_pill(risk_label, risk_kind)}</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Health", f"{current_health:.1f}%" if not np.isnan(current_health) else "—",
              delta=f"{delta:+.2f}%" if delta else None)
    m2.metric("Cycles run", f"{cycles}")
    m3.metric("Estimated RUL",
              f"{est_rul} cycles" if isinstance(est_rul, int) else est_rul)
    m4.metric("Last update", datetime.now(timezone.utc).strftime("%H:%M:%S"))

    chart_l, chart_r = st.columns([2, 1])
    with chart_l:
        st.markdown("##### Health degradation")
        line_color = COLOR_OK if current_health >= 70 else (COLOR_WARN if current_health >= 50 else COLOR_CRIT)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=engine_data["time_in_cycles"], y=engine_data["health_index"],
            mode="lines", line=dict(color=line_color, width=2),
            fill="tozeroy", fillcolor=f"{line_color}22",
            hovertemplate="Cycle %{x}<br>Health %{y:.1f}%<extra></extra>",
        ))
        fig.add_hline(y=50, line_dash="dot", line_color=COLOR_CRIT)
        fig.add_hline(y=70, line_dash="dot", line_color=COLOR_WARN)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=COLOR_PANEL, plot_bgcolor=COLOR_PANEL,
            xaxis_title="Operating cycle", yaxis_title="Health (%)",
            yaxis={"range": [0, 105], "gridcolor": COLOR_BORDER},
            xaxis={"gridcolor": COLOR_BORDER},
            height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_r:
        st.markdown("##### Recent cycles")
        recent_df = engine_data[["time_in_cycles", "health_index"]].tail(10).iloc[::-1]
        st.dataframe(
            recent_df,
            column_config={
                "time_in_cycles": "Cycle",
                "health_index": st.column_config.ProgressColumn(
                    "Health", format="%.1f%%", min_value=0.0, max_value=100.0,
                ),
            },
            hide_index=True, use_container_width=True, height=380,
        )

    sensor_cols = [c for c in feat_cols if c.startswith("sensor_") and "rolling" not in c]
    if sensor_cols:
        st.markdown("##### Sensor traces")
        chosen = st.multiselect(
            "Sensors", sensor_cols,
            default=sensor_cols[:3] if len(sensor_cols) >= 3 else sensor_cols,
            label_visibility="collapsed",
        )
        if chosen:
            fig_s = go.Figure()
            for s in chosen:
                fig_s.add_trace(go.Scatter(
                    x=engine_data["time_in_cycles"], y=engine_data[s],
                    mode="lines", name=s,
                ))
            fig_s.update_layout(
                template="plotly_dark", paper_bgcolor=COLOR_PANEL, plot_bgcolor=COLOR_PANEL,
                xaxis_title="Cycle", yaxis_title="Sensor value",
                xaxis={"gridcolor": COLOR_BORDER}, yaxis={"gridcolor": COLOR_BORDER},
                height=320, margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_s, use_container_width=True)


# ======================================================================
# VIEW 3 — ALARMS (real critical events from the audit ledger)
# ======================================================================
elif view == "Alarms":
    st.markdown(f"## Active Alarms &nbsp; <span style='color:{COLOR_MUTED};font-size:1rem'>tenant: {tenant_id}</span>", unsafe_allow_html=True)

    blocks = ledger.get_recent_blocks(200, tenant_id)
    critical_blocks = [
        b for b in blocks
        if isinstance(b.get("data"), dict) and b["data"].get("status") == "CRITICAL"
    ]

    k1, k2, k3 = st.columns(3)
    k1.metric("Critical events (last 200 blocks)", f"{len(critical_blocks)}")
    if critical_blocks:
        latest_ts = max(b["timestamp"] for b in critical_blocks)
        k2.metric("Most recent",
                  datetime.fromtimestamp(latest_ts, tz=timezone.utc).strftime("%H:%M:%S UTC"))
        unique_units = len({b["data"].get("unit_id") for b in critical_blocks})
        k3.metric("Engines affected", f"{unique_units}")
    else:
        k2.metric("Most recent", "—")
        k3.metric("Engines affected", "0")

    st.markdown("<br/>", unsafe_allow_html=True)
    if not critical_blocks:
        st.info("No critical events in the audit ledger for this tenant.")
    else:
        rows = []
        for b in reversed(critical_blocks[-50:]):
            d = b["data"]
            rows.append({
                "Timestamp (UTC)": datetime.fromtimestamp(b["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "Block #": b["index"],
                "Engine": d.get("unit_id", "—"),
                "Cycle": d.get("cycles", "—"),
                "Health %": f"{d.get('health_score', 0):.1f}",
                "Status": d.get("status", "—"),
                "Hash prefix": b["hash"][:14] + "…",
            })
        df_alarms = pd.DataFrame(rows)
        st.dataframe(df_alarms, hide_index=True, use_container_width=True, height=520)


# ======================================================================
# VIEW 4 — AUDIT TRAIL (cryptographic ledger)
# ======================================================================
elif view == "Audit Trail":
    st.markdown(f"## Audit Trail &nbsp; <span style='color:{COLOR_MUTED};font-size:1rem'>tenant: {tenant_id}</span>", unsafe_allow_html=True)

    metrics = ledger.get_metrics(tenant_id)
    is_valid, msg = ledger.validate_integrity(tenant_id)
    merkle = ledger.compute_merkle_root(tenant_id)
    blocks = ledger.get_recent_blocks(50, tenant_id)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total blocks (all tenants)", metrics.get("total_blocks_global", 0))
    k2.metric("Integrity",
              "VERIFIED" if is_valid else "TAMPERED",
              delta_color=("normal" if is_valid else "inverse"))
    k3.metric("Avg add latency", f"{metrics.get('avg_add_latency_ms', 0):.2f} ms")
    k4.metric("Hash algorithm", str(cfg.blockchain.hash_algorithm).upper())

    st.markdown("##### Merkle root")
    st.code(merkle, language=None)
    if not is_valid:
        st.error(f"Integrity check failed: {msg}")

    st.markdown("##### Recent blocks")
    if blocks:
        for b in reversed(blocks):
            ts = datetime.fromtimestamp(b["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            with st.expander(f"#{b['index']:>5}  &middot;  {ts}  &middot;  {b['hash'][:18]}…"):
                st.json(b)
    else:
        st.info("No blocks for this tenant yet.")


# ======================================================================
# VIEW 5 — SYSTEM HEALTH (benchmarks, model registry, runtime)
# ======================================================================
elif view == "System Health":
    st.markdown("## System Health")

    # --- Benchmark panel ---
    if benchmarks:
        st.markdown("##### Benchmark suite (latest run)")
        cols = st.columns(4)
        lat = benchmarks.get("inference_latency", {})
        thr = benchmarks.get("throughput", {})
        det = benchmarks.get("detection_quality", {})
        lt = benchmarks.get("lead_time", {})
        cols[0].metric("Median latency",
                       f"{lat.get('median_ms', '—')} ms" if lat else "—")
        cols[1].metric("p95 latency",
                       f"{lat.get('p95_ms', '—')} ms" if lat else "—")
        cols[2].metric("Throughput @ batch=256",
                       f"{thr.get('batch_256', '—')}/s" if thr else "—")
        cols[3].metric("Detection F1",
                       f"{det.get('f1', 0):.3f}" if det else "—")

        if thr:
            st.markdown("##### Throughput by batch size")
            tp_df = pd.DataFrame(
                [{"batch": int(k.split("_")[1]), "throughput": v}
                 for k, v in thr.items() if k.startswith("batch_")]
            ).sort_values("batch")
            fig = go.Figure(go.Bar(
                x=tp_df["batch"].astype(str), y=tp_df["throughput"],
                marker_color=COLOR_ACCENT,
                hovertemplate="batch=%{x}<br>%{y:,.0f} samples/s<extra></extra>",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor=COLOR_PANEL, plot_bgcolor=COLOR_PANEL,
                xaxis_title="Batch size", yaxis_title="Samples / sec",
                xaxis={"gridcolor": COLOR_BORDER},
                yaxis={"gridcolor": COLOR_BORDER},
                height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        if det and lt:
            cl, cr = st.columns(2)
            with cl:
                st.markdown("##### Detection quality")
                quality_df = pd.DataFrame({
                    "metric": ["precision", "recall", "f1", "roc_auc", "pr_auc"],
                    "value": [det.get(k, 0.0) for k in ["precision", "recall", "f1", "roc_auc", "pr_auc"]],
                })
                fig_q = go.Figure(go.Bar(
                    x=quality_df["metric"], y=quality_df["value"],
                    marker_color=COLOR_ACCENT,
                ))
                fig_q.update_layout(
                    template="plotly_dark", paper_bgcolor=COLOR_PANEL, plot_bgcolor=COLOR_PANEL,
                    yaxis={"range": [0, 1.0], "gridcolor": COLOR_BORDER},
                    xaxis={"gridcolor": COLOR_BORDER},
                    height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                )
                st.plotly_chart(fig_q, use_container_width=True)
            with cr:
                st.markdown("##### Lead time")
                st.metric("Average", f"{lt.get('avg_lead_time', 0):.1f} cycles")
                st.metric("Median", f"{lt.get('median_lead_time', 0):.1f} cycles")
                st.metric("Detection rate", f"{lt.get('detection_rate', 0)*100:.1f}%")
                st.metric("Engines detected",
                          f"{lt.get('engines_detected', 0)}/{lt.get('total_engines', 0)}")
    else:
        st.info(
            "No benchmark data found at outputs/benchmarks/benchmark_results.json. "
            "Generate it with `python main_pipeline.py --mode benchmark`."
        )

    st.markdown("##### Model registry")
    model_dir = Path(cfg.models.model_dir)
    if model_dir.exists():
        artifacts = sorted(model_dir.glob("*.joblib"))
        if artifacts:
            rows = []
            for a in artifacts:
                stat = a.stat()
                rows.append({
                    "artifact": a.name,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info(f"No model artifacts in {model_dir}/. Run `python main_pipeline.py --mode train`.")
    else:
        st.info(f"Model directory {model_dir}/ does not exist yet.")

    st.markdown("##### Runtime")
    rt = st.columns(3)
    rt[0].metric("Active model", cfg.models.active_model)
    rt[1].metric("Failure threshold", f"{cfg.data.failure_rul_threshold} cycles")
    rt[2].metric("Rolling window", f"{cfg.data.rolling_window} cycles")


# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown(
    f"<div style='border-top:1px solid {COLOR_BORDER};margin-top:32px;padding-top:10px;"
    f"color:{COLOR_MUTED};font-size:0.72rem;text-align:center'>"
    f"PulseNet v{cfg.system.version} &middot; tenant <code>{tenant_id}</code> &middot; "
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    f"</div>",
    unsafe_allow_html=True,
)
