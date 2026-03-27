"""
dashboard.py
------------
Interactive Streamlit dashboard for real-time predictive maintenance monitoring.

Features:
- Live sensor feed simulation per machine
- Real-time anomaly score gauge
- RUL countdown per machine
- Alert log with severity levels (warning / critical)
- Historical fault timeline

Usage:
    streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
from datetime import datetime, timedelta

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #2d2d44;
        text-align: center;
    }
    .alert-critical {
        background: rgba(255, 75, 75, 0.15);
        border-left: 4px solid #ff4b4b;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px 0;
    }
    .alert-warning {
        background: rgba(255, 165, 0, 0.15);
        border-left: 4px solid #ffa500;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px 0;
    }
    .alert-normal {
        background: rgba(0, 200, 100, 0.1);
        border-left: 4px solid #00c864;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Simulated Live Data ───────────────────────────────────────────────────────

MACHINES = ["M01", "M02", "M03"]
SENSORS = ["temperature", "vibration", "pressure", "rpm", "current", "torque", "oil_level", "acoustics"]

NOMINAL = {
    "temperature": 75.0, "vibration": 0.3, "pressure": 12.0,
    "rpm": 1480.0, "current": 18.0, "torque": 95.0,
    "oil_level": 90.0, "acoustics": 62.0
}
SENSOR_UNITS = {
    "temperature": "°C", "vibration": "g", "pressure": "bar",
    "rpm": "RPM", "current": "A", "torque": "Nm",
    "oil_level": "%", "acoustics": "dB"
}


def simulate_reading(health: float, noise: float = 0.03) -> dict:
    """Generate a single sensor reading for a given health level."""
    sensitivity = {
        "temperature": +15.0, "vibration": +0.8, "pressure": -3.0,
        "rpm": -50.0, "current": +5.0, "torque": +12.0,
        "oil_level": -15.0, "acoustics": +12.0
    }
    degradation = 1.0 - health
    return {
        ch: NOMINAL[ch] + sensitivity[ch] * degradation + np.random.normal(0, abs(NOMINAL[ch]) * noise)
        for ch in SENSORS
    }


def get_alert_level(anomaly_score: float) -> str:
    if anomaly_score > 0.75:
        return "critical"
    elif anomaly_score > 0.45:
        return "warning"
    return "normal"


# ─── Session State Init ───────────────────────────────────────────────────────

if "machine_states" not in st.session_state:
    st.session_state.machine_states = {
        m: {"health": np.random.uniform(0.6, 1.0), "rul": np.random.uniform(50, 200)}
        for m in MACHINES
    }

if "history" not in st.session_state:
    st.session_state.history = {m: [] for m in MACHINES}

if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "tick" not in st.session_state:
    st.session_state.tick = 0


def update_states():
    """Advance simulation by one tick."""
    st.session_state.tick += 1
    now = datetime.now()

    for m in MACHINES:
        state = st.session_state.machine_states[m]

        # Degrade health
        state["health"] = max(0.0, state["health"] - np.random.uniform(0.003, 0.012))
        state["rul"] = max(0.0, state["rul"] - np.random.uniform(0.5, 2.0))

        # Randomly reset (simulate repair)
        if state["health"] < 0.1 or state["rul"] < 2:
            state["health"] = 1.0
            state["rul"] = np.random.uniform(100, 250)

        # Sensor reading
        reading = simulate_reading(state["health"])
        anomaly_score = max(0, min(1, 1 - state["health"] + np.random.normal(0, 0.05)))
        level = get_alert_level(anomaly_score)

        record = {
            "timestamp": now,
            "health": state["health"],
            "rul": state["rul"],
            "anomaly_score": anomaly_score,
            "alert_level": level,
            **reading
        }
        st.session_state.history[m].append(record)

        # Keep last 200 records
        if len(st.session_state.history[m]) > 200:
            st.session_state.history[m] = st.session_state.history[m][-200:]

        # Add alert if not normal
        if level != "normal":
            st.session_state.alerts.insert(0, {
                "time": now.strftime("%H:%M:%S"),
                "machine": m,
                "level": level,
                "score": round(anomaly_score, 3),
                "rul": round(state["rul"], 1)
            })

    # Keep last 50 alerts
    st.session_state.alerts = st.session_state.alerts[:50]


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔧 PdM Dashboard")
    st.markdown("---")

    auto_refresh = st.toggle("Auto Refresh", value=True)
    refresh_rate = st.slider("Refresh rate (s)", 1, 10, 2)

    st.markdown("---")
    st.markdown("**Machine Selection**")
    selected_machine = st.selectbox("Detail view for:", MACHINES)

    st.markdown("---")
    st.markdown("**Thresholds**")
    warn_thresh = st.slider("Warning threshold", 0.3, 0.7, 0.45)
    crit_thresh = st.slider("Critical threshold", 0.5, 0.95, 0.75)

    if st.button("🔄 Force Refresh"):
        update_states()
        st.rerun()


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🏭 Predictive Maintenance — Real-Time Dashboard")
st.markdown(f"`Tick #{st.session_state.tick}` · Last update: `{datetime.now().strftime('%H:%M:%S')}`")
st.markdown("---")

# Update state
update_states()

# ─── KPI Row ──────────────────────────────────────────────────────────────────

kpi_cols = st.columns(len(MACHINES) + 1)

total_alerts = len([a for a in st.session_state.alerts if a["level"] == "critical"])
with kpi_cols[0]:
    st.metric("🚨 Active Critical Alerts", total_alerts)

for i, m in enumerate(MACHINES):
    state = st.session_state.machine_states[m]
    history = st.session_state.history[m]
    score = history[-1]["anomaly_score"] if history else 0
    level = get_alert_level(score)
    icon = "🔴" if level == "critical" else "🟡" if level == "warning" else "🟢"

    with kpi_cols[i + 1]:
        st.metric(
            f"{icon} Machine {m}",
            f"Health: {state['health']:.0%}",
            f"RUL: {state['rul']:.0f}h"
        )

st.markdown("---")

# ─── Anomaly Score Gauges ─────────────────────────────────────────────────────

st.subheader("🎯 Anomaly Scores — Live")
gauge_cols = st.columns(len(MACHINES))

for i, m in enumerate(MACHINES):
    history = st.session_state.history[m]
    score = history[-1]["anomaly_score"] if history else 0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        title={"text": f"Machine {m}", "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": "#ff4b4b" if score > 0.75 else "#ffa500" if score > 0.45 else "#00c864"},
            "bgcolor": "#1a1a2e",
            "bordercolor": "#2d2d44",
            "steps": [
                {"range": [0, 45], "color": "rgba(0,200,100,0.15)"},
                {"range": [45, 75], "color": "rgba(255,165,0,0.15)"},
                {"range": [75, 100], "color": "rgba(255,75,75,0.2)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": crit_thresh * 100
            }
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )
    gauge_cols[i].plotly_chart(fig, use_container_width=True)

# ─── Sensor Time Series (selected machine) ────────────────────────────────────

st.subheader(f"📡 Sensor Feed — Machine {selected_machine}")

history = st.session_state.history[selected_machine]
if len(history) > 1:
    df_hist = pd.DataFrame(history)

    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f"{s} ({SENSOR_UNITS[s]})" for s in SENSORS],
        vertical_spacing=0.18
    )
    colors = px.colors.qualitative.Plotly

    for idx, sensor in enumerate(SENSORS):
        row = idx // 4 + 1
        col = idx % 4 + 1
        fig.add_trace(
            go.Scatter(
                x=df_hist["timestamp"],
                y=df_hist[sensor],
                mode="lines",
                line=dict(color=colors[idx % len(colors)], width=1.5),
                name=sensor,
                showlegend=False
            ),
            row=row, col=col
        )
        # Nominal reference line
        fig.add_hline(
            y=NOMINAL[sensor],
            line_dash="dot",
            line_color="rgba(255,255,255,0.3)",
            row=row, col=col
        )

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,26,46,0.8)",
        font={"color": "white", "size": 10},
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=8))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=8))

    st.plotly_chart(fig, use_container_width=True)

# ─── RUL Timeline ─────────────────────────────────────────────────────────────

st.subheader("⏳ Remaining Useful Life (RUL) — All Machines")

if all(len(st.session_state.history[m]) > 1 for m in MACHINES):
    fig_rul = go.Figure()
    rul_colors = {"M01": "#00d4ff", "M02": "#ff6b9d", "M03": "#ffd700"}

    for m in MACHINES:
        df_m = pd.DataFrame(st.session_state.history[m])
        fig_rul.add_trace(go.Scatter(
            x=df_m["timestamp"],
            y=df_m["rul"],
            name=f"Machine {m}",
            mode="lines",
            line=dict(color=rul_colors[m], width=2)
        ))

    fig_rul.add_hline(y=24, line_dash="dash", line_color="#ff4b4b",
                      annotation_text="Critical threshold (24h)", annotation_font_color="#ff4b4b")
    fig_rul.add_hline(y=48, line_dash="dot", line_color="#ffa500",
                      annotation_text="Warning threshold (48h)", annotation_font_color="#ffa500")

    fig_rul.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,26,46,0.8)",
        font={"color": "white"},
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="RUL (hours)"),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_rul, use_container_width=True)

# ─── Alert Log ────────────────────────────────────────────────────────────────

st.subheader("🚨 Alert Log")

if st.session_state.alerts:
    for alert in st.session_state.alerts[:15]:
        css_class = f"alert-{alert['level']}"
        icon = "🔴" if alert["level"] == "critical" else "🟡"
        st.markdown(
            f'<div class="{css_class}">'
            f'{icon} <b>[{alert["time"]}]</b> Machine <b>{alert["machine"]}</b> — '
            f'{alert["level"].upper()} | Score: {alert["score"]} | RUL: {alert["rul"]}h'
            f'</div>',
            unsafe_allow_html=True
        )
else:
    st.success("✅ No active alerts — all machines operating normally")

# ─── Auto Refresh ─────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
