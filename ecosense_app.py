# =============================================================================
# EcoSense v2.0 – Streamlit Interactive Dashboard
# Jalankan: streamlit run ecosense_app.py
#
# Fitur:
#   - Dashboard Real-time: Energy stream, SPC, Anomaly, Model metrics
#   - Optimizer Interaktif: Slider real-time → langsung tampil estimasi energi
#   - Multi-Solusi: 5 skenario optimal (tidak hanya 1 rekomendasi)
#   - Pareto Explorer: Scatter plot Pareto front yang bisa di-hover
#   - Stress Test: Simulasi kenaikan produksi langsung di UI
#   - Alert Monitor: Stream alert real-time dengan notifikasi
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from pathlib import Path

# ── import engine dari file v2 ──
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ecosense_iot_optimization_v2 import (
    generate_iot_sensor_data,
    build_energy_model,
    detect_sensor_anomalies,
    calculate_control_limits,
    optimize_machine_settings,
    stress_test_simulation,
    simulate_streaming,
    predict_energy,
    CFG,
)

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EcoSense v2.0 – IoT Energy Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS dark industrial theme ──
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .block-container { padding-top: 1rem; }
    div[data-testid="metric-container"] {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 12px 16px;
    }
    div[data-testid="metric-container"] label { color: #8b949e !important; }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #4fc3f7 !important; font-size: 1.6rem !important;
    }
    .solution-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 16px; margin: 8px 0;
    }
    .solution-best { border-color: #69f0ae !important; }
    .solution-energy { border-color: #4fc3f7 !important; }
    .solution-balanced { border-color: #ffb74d !important; }
    .alert-critical { background:#3d1c1c; border-left:4px solid #ff5252; padding:8px 12px; border-radius:4px; margin:4px 0; }
    .alert-warning  { background:#3d2f1c; border-left:4px solid #ffb74d; padding:8px 12px; border-radius:4px; margin:4px 0; }
    .alert-info     { background:#1c2d3d; border-left:4px solid #4fc3f7; padding:8px 12px; border-radius:4px; margin:4px 0; }
    h1,h2,h3 { color: #e6edf3 !important; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #4fc3f7 !important; }
    .stSlider label { color: #8b949e !important; }
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE & CACHE
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="⚙️ Membangun dataset IoT sensor (7 hari × 3 mesin)…")
def load_iot_data():
    return generate_iot_sensor_data()

@st.cache_data(show_spinner="🔬 Melatih Ensemble Energy Model (Ridge+RF+HGB)…")
def load_model(_df, machine_id):
    return build_energy_model(_df, machine_id=machine_id)

@st.cache_data(show_spinner="🔎 Menjalankan Anomaly Ensemble (IForest+LOF+Z-Score)…")
def load_anomalies(_df, machine_id):
    return detect_sensor_anomalies(_df, machine_id=machine_id)

@st.cache_data(show_spinner="📊 Menghitung batas kontrol SPC (X-bar + CUSUM + EWMA)…")
def load_spc(_df, machine_id):
    energy = _df[_df["machine_id"] == machine_id]["Y_energy_kWh"].values
    return calculate_control_limits(energy)

@st.cache_data(show_spinner="🎯 Menjalankan Multi-Objective Optimization (NSGA-II)…")
def load_optimization(_model):
    return optimize_machine_settings(_model)

@st.cache_data(show_spinner="⚡ Simulasi stress test (7 skenario)…")
def load_stress(_model):
    return stress_test_simulation(_model)


def get_or_init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ EcoSense v2.0")
    st.markdown("*IoT Energy Optimization Twin*")
    st.divider()

    machine_id = st.selectbox("🏭 Target Mesin", ["M01", "M02", "M03"],
                               help="Pilih mesin yang ingin dianalisis")
    st.divider()

    st.markdown("### ⚙️ Parameter Simulasi")
    tariff = st.number_input("Tarif Energi (Rp/kWh)", value=1500,
                              min_value=500, max_value=5000, step=100)
    annual_hours = st.number_input("Jam Operasi/Tahun", value=8000,
                                    min_value=1000, max_value=8760, step=100)
    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**v2.0** | AHM Manufacturing")
    st.markdown("*Ensemble ML · NSGA-II · CUSUM/EWMA*")


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════

df          = load_iot_data()
model_r     = load_model(df, machine_id)
anom_df, anom_summary = load_anomalies(df, machine_id)
spc_r       = load_spc(df, machine_id)
opt_r       = load_optimization(model_r)
stress_df   = load_stress(model_r)

mdf = anom_df[anom_df["machine_id"] == machine_id].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════

st.markdown(f"# ⚡ EcoSense — IoT Energy Optimization Digital Twin")
st.markdown(f"**Mesin aktif: `{machine_id}`** | Data: 7 hari × {len(mdf):,} titik | "
            f"Model R²: `{model_r['r2_test']:.4f}`")

# ── KPI Row ──
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📊 Total Data Points", f"{len(mdf):,}")
k2.metric("🚨 Anomali (Ensemble)", f"{anom_summary['count']}",
          delta=f"F1={anom_summary['f1']:.3f}", delta_color="off")
k3.metric("⚡ Hemat Energi", f"{opt_r['saving_pct']:.1f}%",
          delta=f"Rp {opt_r['annual_saving_idr']/1e9:.2f}M/thn")
k4.metric("🤖 Model R²", f"{model_r['r2_test']:.4f}",
          delta=f"MAE={model_r['mae_test']:.3f}", delta_color="off")
k5.metric("⚠️ CUSUM OOC", f"{int(spc_r['cusum_ooc'].sum())}",
          delta=f"EWMA: {int(spc_r['ewma_ooc'].sum())}", delta_color="inverse")

st.divider()


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Dashboard Real-time",
    "🎯 Optimizer Interaktif",
    "📊 Multi-Solusi Optimal",
    "🔬 Analisis Model",
    "⚡ Stress Test",
    "🚨 Alert Monitor",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 – DASHBOARD REAL-TIME
# ══════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### 📡 Monitoring Energi Real-time")

    # Pilih rentang tampilan
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        n_disp = st.slider("Jumlah titik ditampilkan", 100, len(mdf),
                           min(600, len(mdf)), step=50, key="ndisp")
    with col_ctrl2:
        show_anomaly = st.toggle("Tampilkan Anomali", value=True)

    e_disp = mdf["Y_energy_kWh"].values[-n_disp:]
    t_disp = mdf["timestamp"].values[-n_disp:]
    a_disp = mdf["anomaly_ensemble"].values[-n_disp:]

    # Energy Stream Chart
    fig_stream = go.Figure()
    fig_stream.add_trace(go.Scatter(
        x=t_disp, y=e_disp, mode="lines", name="Energy (kWh)",
        line=dict(color="#4fc3f7", width=1), opacity=0.9,
    ))
    if show_anomaly:
        anom_mask = a_disp.astype(bool)
        fig_stream.add_trace(go.Scatter(
            x=t_disp[anom_mask], y=e_disp[anom_mask],
            mode="markers", name="Anomali Ensemble",
            marker=dict(color="#ff5252", size=7, symbol="x"),
        ))
    fig_stream.add_hline(y=spc_r["robust_upper"], line_dash="dash",
                          line_color="#ffb74d", annotation_text="UCL")
    fig_stream.add_hline(y=spc_r["robust_lower"], line_dash="dash",
                          line_color="#ffb74d", annotation_text="LCL")
    fig_stream.update_layout(
        template="plotly_dark", height=320, margin=dict(t=30, b=30),
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        title=f"Energy Stream – {machine_id} (Last {n_disp} readings)",
    )
    st.plotly_chart(fig_stream, use_container_width=True)

    # SPC Charts row
    col_c, col_e = st.columns(2)

    with col_c:
        fig_cusum = go.Figure()
        cx = np.arange(len(spc_r["cusum_pos"]))
        fig_cusum.add_trace(go.Scatter(x=cx, y=spc_r["cusum_pos"],
                                        mode="lines", name="C⁺",
                                        line=dict(color="#69f0ae", width=1)))
        fig_cusum.add_trace(go.Scatter(x=cx, y=-spc_r["cusum_neg"],
                                        mode="lines", name="C⁻",
                                        line=dict(color="#ff5252", width=1)))
        fig_cusum.add_hline(y= spc_r["cusum_threshold"], line_dash="dot",
                             line_color="#ffb74d", annotation_text="H")
        fig_cusum.add_hline(y=-spc_r["cusum_threshold"], line_dash="dot",
                             line_color="#ffb74d")
        fig_cusum.update_layout(template="plotly_dark", height=260,
                                 paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                 margin=dict(t=40, b=20),
                                 title="CUSUM Chart (OOC: " +
                                       str(int(spc_r["cusum_ooc"].sum())) + ")")
        st.plotly_chart(fig_cusum, use_container_width=True)

    with col_e:
        fig_ewma = go.Figure()
        ex = np.arange(len(spc_r["ewma"]))
        fig_ewma.add_trace(go.Scatter(x=ex, y=spc_r["ewma"],
                                       mode="lines", name="EWMA",
                                       line=dict(color="#ce93d8", width=1)))
        fig_ewma.add_hline(y=spc_r["UCL_ewma"], line_dash="dot",
                            line_color="#ff5252", annotation_text="UCL")
        fig_ewma.add_hline(y=spc_r["LCL_ewma"], line_dash="dot",
                            line_color="#4db6ac", annotation_text="LCL")
        ewma_ooc_idx = np.where(spc_r["ewma_ooc"])[0]
        if len(ewma_ooc_idx):
            fig_ewma.add_trace(go.Scatter(
                x=ewma_ooc_idx, y=spc_r["ewma"][ewma_ooc_idx],
                mode="markers", name="OOC",
                marker=dict(color="#ff5252", size=5),
            ))
        fig_ewma.update_layout(template="plotly_dark", height=260,
                                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                margin=dict(t=40, b=20),
                                title="EWMA Chart (OOC: " +
                                      str(int(spc_r["ewma_ooc"].sum())) + ")")
        st.plotly_chart(fig_ewma, use_container_width=True)

    # Anomaly breakdown bar chart
    st.markdown("#### Perbandingan Detektor Anomali")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        detectors = ["IsolationForest", "LOF", "Z-Score MAD", "Ensemble (≥2/3)"]
        counts = [
            int(mdf["anomaly_iforest"].sum()),
            int(mdf["anomaly_lof"].sum()),
            int(mdf["anomaly_zscore"].sum()),
            int(mdf["anomaly_ensemble"].sum()),
        ]
        colors = ["#4fc3f7", "#ce93d8", "#4db6ac", "#ff5252"]
        fig_adet = go.Figure(go.Bar(
            x=detectors, y=counts, marker_color=colors,
            text=counts, textposition="outside",
        ))
        fig_adet.update_layout(template="plotly_dark", height=260,
                                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                margin=dict(t=20, b=20),
                                showlegend=False)
        st.plotly_chart(fig_adet, use_container_width=True)

    with col_b:
        st.markdown("**Evaluasi vs Ground Truth**")
        st.metric("Precision", f"{anom_summary['precision']:.3f}")
        st.metric("Recall",    f"{anom_summary['recall']:.3f}")
        st.metric("F1-Score",  f"{anom_summary['f1']:.3f}")

    # Shift energy profile
    st.markdown("#### Profil Energi per Shift × Mesin")
    pivot = df.groupby(["machine_id", "shift"])["Y_energy_kWh"].mean().reset_index()
    fig_shift = px.bar(
        pivot, x="machine_id", y="Y_energy_kWh", color="shift",
        barmode="group", color_discrete_sequence=["#4fc3f7", "#ffb74d", "#ce93d8"],
        labels={"Y_energy_kWh": "Rata-rata kWh", "machine_id": "Mesin",
                "shift": "Shift"},
        template="plotly_dark", height=280,
    )
    fig_shift.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                             margin=dict(t=20, b=20))
    st.plotly_chart(fig_shift, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 – OPTIMIZER INTERAKTIF
# ══════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 🎯 Optimizer Interaktif – Setting Mesin Real-time")
    st.markdown(
        "Geser slider di bawah untuk melihat estimasi konsumsi energi secara langsung. "
        "Warna indikator berubah sesuai tingkat efisiensi."
    )

    col_s, col_r = st.columns([1, 1])

    with col_s:
        st.markdown("#### ⚙️ Parameter Mesin")
        temp_v  = st.slider("🌡️ Suhu Mesin (°C)",     55, 100, 70, key="opt_temp")
        load_v  = st.slider("⚙️ Beban Mesin (%)",      30, 100, 75, key="opt_load")
        rpm_v   = st.slider("🔄 Kecepatan RPM",       800, 2200, 1400, step=50, key="opt_rpm")
        amb_v   = st.slider("🌤️ Suhu Ambient (°C)",    20, 42, 30, key="opt_amb")
        hum_v   = st.slider("💧 Kelembaban (%)",        40, 95, 70, key="opt_hum")
        eff_v   = st.slider("📈 Efisiensi Mesin",      0.88, 1.02, 0.97,
                             step=0.01, key="opt_eff", format="%.2f")

    with col_r:
        # Hitung real-time
        current_energy = predict_energy(model_r, {
            "X_Temp": temp_v, "X_Load": load_v, "X_RPM": rpm_v,
            "X_Ambient_Temp": amb_v, "X_Humidity": hum_v, "X_Efficiency": eff_v,
        })
        baseline_energy = predict_energy(model_r, {
            "X_Temp": 72, "X_Load": 75, "X_RPM": 1500,
            "X_Ambient_Temp": 30, "X_Humidity": 70, "X_Efficiency": 0.97,
        })
        opt_energy = opt_r["best_point"]["energy"]

        saving_vs_baseline = (baseline_energy - current_energy) / baseline_energy * 100
        annual_saving = (baseline_energy - current_energy) * annual_hours * tariff

        # Warna gauge berdasarkan efisiensi
        if current_energy < opt_energy * 1.05:
            gauge_color = "#69f0ae"; status = "✅ OPTIMAL"
        elif current_energy < baseline_energy:
            gauge_color = "#ffb74d"; status = "🟡 LEBIH BAIK"
        elif current_energy < baseline_energy * 1.1:
            gauge_color = "#ff9800"; status = "🟠 NORMAL"
        else:
            gauge_color = "#ff5252"; status = "🔴 BOROS"

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_energy,
            delta={"reference": baseline_energy, "valueformat": ".3f",
                   "suffix": " kWh"},
            gauge={
                "axis": {"range": [opt_energy * 0.9, baseline_energy * 1.3],
                          "tickcolor": "#8b949e"},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [opt_energy * 0.9, opt_energy * 1.05],
                     "color": "#1a3a2a"},
                    {"range": [opt_energy * 1.05, baseline_energy],
                     "color": "#2a2a1a"},
                    {"range": [baseline_energy, baseline_energy * 1.3],
                     "color": "#3a1a1a"},
                ],
                "threshold": {
                    "line": {"color": "#ff5252", "width": 2},
                    "thickness": 0.75, "value": baseline_energy,
                },
            },
            title={"text": f"Konsumsi Energi Saat Ini<br>{status}",
                   "font": {"color": "#e6edf3", "size": 13}},
            number={"suffix": " kWh", "font": {"size": 36, "color": gauge_color}},
        ))
        fig_gauge.update_layout(
            template="plotly_dark", height=300,
            paper_bgcolor="#0d1117",
            font={"color": "#e6edf3"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("#### 📊 Perbandingan Instan")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Saat Ini", f"{current_energy:.3f} kWh")
        rc2.metric("Baseline", f"{baseline_energy:.3f} kWh",
                   delta=f"{saving_vs_baseline:+.1f}%")
        rc3.metric("Target Optimal", f"{opt_energy:.3f} kWh",
                   delta=f"{(current_energy - opt_energy):+.3f} kWh",
                   delta_color="inverse")

        if annual_saving > 0:
            st.success(f"💰 Estimasi hemat vs baseline: **Rp {annual_saving/1e6:.1f} Juta/tahun**")
        else:
            st.warning(f"📈 Setting ini lebih boros Rp {-annual_saving/1e6:.1f} Juta/tahun vs baseline")

    # Radar chart: profil mesin saat ini vs optimal
    st.markdown("---")
    st.markdown("#### 🕸️ Radar Chart: Profil Mesin Saat Ini vs Optimal")

    categories = ["Suhu (%max)", "Beban (%max)", "RPM (%max)",
                  "Ambient (%max)", "Humidity (%max)", "Efisiensi (%)"]
    current_norm = [temp_v/100, load_v/100, rpm_v/2200,
                    amb_v/42, hum_v/95, eff_v]
    optimal_norm = [
        opt_r["best_point"]["temp"] / 100,
        opt_r["best_point"]["load"] / 100,
        opt_r["best_point"]["rpm"]  / 2200,
        30/42, 70/95, 0.97,
    ]

    fig_radar = go.Figure()
    for vals, name, color in [
        (current_norm, "Setting Saat Ini", "#ffb74d"),
        (optimal_norm, "Rekomendasi Optimal", "#69f0ae"),
    ]:
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill="toself", name=name,
            line=dict(color=color), opacity=0.6,
        ))
    fig_radar.update_layout(
        polar=dict(bgcolor="#161b22",
                   radialaxis=dict(visible=True, range=[0, 1],
                                   gridcolor="#30363d", color="#8b949e"),
                   angularaxis=dict(gridcolor="#30363d", color="#8b949e")),
        template="plotly_dark", height=360,
        paper_bgcolor="#0d1117",
        legend=dict(x=0.85, y=1.1),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3 – MULTI-SOLUSI OPTIMAL
# ══════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 📊 5 Solusi Optimal — Bukan Hanya 1 Rekomendasi")
    st.markdown(
        "Sistem mengidentifikasi **5 skenario berbeda** dengan trade-off energi vs produksi "
        "yang bisa dipilih sesuai prioritas manajer operasi."
    )

    # Bangun 5 solusi dari Pareto front
    pareto = opt_r["pareto_front"]
    if len(pareto) >= 5:
        pareto_arr = np.array([(p[0], p[1], p[2], p[3], p[4]) for p in pareto])
    else:
        # Jika Pareto < 5, buat variasi manual
        bp = opt_r["best_point"]
        pareto_arr = np.array([
            [bp["temp"],       bp["load"],       bp["rpm"],       bp["energy"],       bp["production_index"]],
            [bp["temp"]+5,     min(bp["load"]+10, 95), min(bp["rpm"]+100, 1900), bp["energy"]*1.04, bp["production_index"]*1.05],
            [bp["temp"]+10,    min(bp["load"]+15, 95), min(bp["rpm"]+200, 1900), bp["energy"]*1.08, bp["production_index"]*1.10],
            [max(bp["temp"]-5, 55), max(bp["load"]-5, 50), max(bp["rpm"]-100, 1000), bp["energy"]*0.97, bp["production_index"]*0.95],
            [72.0,             75.0,             1500.0,          opt_r["baseline_energy"], 75*1500/1e5],
        ])

    # Sort by energy ascending
    pareto_arr = pareto_arr[pareto_arr[:, 3].argsort()]

    # Pilih 5 representatif: min-energy, max-production, 3 balanced
    prod_col = pareto_arr[:, 4]
    ener_col = pareto_arr[:, 3]
    prod_max = prod_col.max()

    idx_min_e  = 0  # minimum energy (already sorted)
    idx_max_p  = int(prod_col.argmax())
    idx_bal1   = int(len(pareto_arr) // 4)
    idx_bal2   = int(len(pareto_arr) // 2)
    idx_bal3   = int(3 * len(pareto_arr) // 4)

    solution_indices = [idx_min_e, idx_bal1, idx_bal2, idx_bal3, idx_max_p]
    solution_labels = [
        ("🟢 Hemat Energi Maksimal", "Prioritas: Efisiensi energi tertinggi. Produksi sedikit turun.", "solution-energy"),
        ("🔵 Seimbang I", "Keseimbangan baik antara penghematan energi dan output produksi.", "solution-balanced"),
        ("🟡 Seimbang II", "Titik tengah Pareto. Rekomendasi default untuk operasi normal.", "solution-balanced"),
        ("🟠 Seimbang III", "Produksi lebih tinggi dengan pengorbanan efisiensi energi moderat.", "solution-balanced"),
        ("🔴 Produksi Maksimal", "Prioritas: Output produksi tertinggi. Konsumsi energi lebih besar.", "solution-best"),
    ]

    baseline_e = opt_r["baseline_energy"]

    # Tampilkan 5 solusi
    for i, (sol_idx, (title, desc, css_class)) in enumerate(
        zip(solution_indices, solution_labels)
    ):
        row = pareto_arr[sol_idx]
        temp_s, load_s, rpm_s, energy_s, prod_s = row
        saving_pct = (baseline_e - energy_s) / baseline_e * 100
        annual_idr = (baseline_e - energy_s) * annual_hours * tariff

        with st.expander(f"{title}  —  ⚡ {energy_s:.3f} kWh  |  📦 Prod.Index: {prod_s:.3f}",
                          expanded=(i == 2)):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**⚙️ Setting Mesin**")
                st.markdown(f"- Suhu: **{temp_s:.1f} °C**")
                st.markdown(f"- Beban: **{load_s:.1f} %**")
                st.markdown(f"- RPM: **{rpm_s:.0f}**")
            with c2:
                st.markdown("**📊 Estimasi Output**")
                st.metric("Energi/jam", f"{energy_s:.3f} kWh",
                           delta=f"{saving_pct:+.1f}% vs baseline",
                           delta_color="inverse" if saving_pct >= 0 else "normal")
                st.metric("Production Index", f"{prod_s:.4f}")
            with c3:
                st.markdown("**💰 Dampak Finansial**")
                if annual_idr > 0:
                    st.success(f"Hemat **Rp {annual_idr/1e6:.1f} Juta/thn**")
                else:
                    st.warning(f"Lebih mahal Rp {-annual_idr/1e6:.1f} Juta/thn")
                st.markdown(f"*{desc}*")

            # Tombol terapkan ke simulator
            if st.button(f"↗️ Terapkan ke Optimizer", key=f"apply_sol_{i}"):
                st.session_state["opt_temp"] = float(np.clip(temp_s, 55, 100))
                st.session_state["opt_load"] = float(np.clip(load_s, 30, 100))
                st.session_state["opt_rpm"]  = int(np.clip(rpm_s, 800, 2200))
                st.info("✅ Setting diterapkan! Buka tab **Optimizer Interaktif**.")

    st.divider()

    # Pareto Front Explorer
    st.markdown("#### 🗺️ Pareto Front Explorer — Hover untuk detail")
    if len(pareto) > 5:
        pf_df = pd.DataFrame(pareto, columns=["Temp", "Load", "RPM", "Energy", "ProdIndex"])
        # Highlight 5 solusi terpilih
        highlight = np.zeros(len(pf_df), dtype=bool)
        highlight[solution_indices] = True
        pf_df["highlight"] = np.where(highlight, "Solusi Pilihan", "Kandidat")

        fig_pareto = px.scatter(
            pf_df, x="Energy", y="ProdIndex",
            color="highlight",
            color_discrete_map={"Solusi Pilihan": "#ff5252", "Kandidat": "#4fc3f7"},
            hover_data={"Temp": ":.1f", "Load": ":.1f", "RPM": ":.0f",
                        "Energy": ":.3f", "ProdIndex": ":.4f"},
            labels={"Energy": "Konsumsi Energi (kWh)", "ProdIndex": "Production Index"},
            template="plotly_dark", height=350,
            title="Pareto Front: Energi (min) vs Produksi (max)",
        )
        fig_pareto.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22")
        st.plotly_chart(fig_pareto, use_container_width=True)

    # Tabel perbandingan
    st.markdown("#### 📋 Tabel Perbandingan Lengkap")
    comparison_data = []
    for i, (sol_idx, (title, _, _)) in enumerate(zip(solution_indices, solution_labels)):
        row = pareto_arr[sol_idx]
        saving_pct = (baseline_e - row[3]) / baseline_e * 100
        comparison_data.append({
            "Skenario": title.split(" ", 1)[1],
            "Suhu (°C)": f"{row[0]:.1f}",
            "Beban (%)": f"{row[1]:.1f}",
            "RPM": f"{row[2]:.0f}",
            "Energi (kWh)": f"{row[3]:.3f}",
            "Hemat (%)": f"{saving_pct:+.1f}%",
            "Hemat/thn (Rp Juta)": f"{(baseline_e - row[3]) * annual_hours * tariff / 1e6:.1f}",
            "Prod. Index": f"{row[4]:.4f}",
        })
    comp_df = pd.DataFrame(comparison_data)

    # Highlight baris dengan hemat terbesar
    def highlight_best(row):
        pct = float(row["Hemat (%)"].rstrip("%"))
        if pct == max(float(r["Hemat (%)"].rstrip("%")) for r in comparison_data):
            return ["background-color: #1a3a2a"] * len(row)
        return [""] * len(row)

    st.dataframe(comp_df.style.apply(highlight_best, axis=1),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 – ANALISIS MODEL
# ══════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### 🔬 Analisis Ensemble Energy Model")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        # Actual vs Predicted scatter
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=model_r["y_test"], y=model_r["y_pred_test"],
            mode="markers", name="Test Points",
            marker=dict(color="#4fc3f7", size=3, opacity=0.4),
        ))
        lims = [
            min(model_r["y_test"].min(), model_r["y_pred_test"].min()),
            max(model_r["y_test"].max(), model_r["y_pred_test"].max()),
        ]
        fig_avp.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                      name=f"Perfect (R²={model_r['r2_test']:.4f})",
                                      line=dict(color="#69f0ae", width=2)))
        fig_avp.update_layout(
            template="plotly_dark", height=340,
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            title="Aktual vs Prediksi (Ensemble)",
            xaxis_title="Aktual (kWh)", yaxis_title="Prediksi (kWh)",
        )
        st.plotly_chart(fig_avp, use_container_width=True)

    with col_m2:
        # Permutation Feature Importance
        imp = model_r["importance"]
        sorted_imp = sorted(imp.items(), key=lambda x: abs(x[1]), reverse=True)
        fi_df = pd.DataFrame(sorted_imp, columns=["Feature", "Importance"])
        fi_df["Color"] = fi_df["Importance"].apply(
            lambda v: "#ff5252" if v < 0 else "#69f0ae")

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h", marker_color=fi_df["Color"],
            text=[f"{v:+.4f}" for v in fi_df["Importance"]],
            textposition="outside",
        ))
        fig_fi.update_layout(
            template="plotly_dark", height=340,
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            title="Feature Importance (Permutation)",
            xaxis_title="Importance Score", yaxis_title="",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Residual plot
    fig_res = go.Figure()
    res = model_r["residuals"]
    fig_res.add_trace(go.Scatter(
        x=model_r["y_pred_test"], y=res,
        mode="markers", name="Residual",
        marker=dict(color="#ffb74d", size=3, opacity=0.4),
    ))
    fig_res.add_hline(y=0, line_color="#69f0ae", line_width=1.5)
    for sigma_mult in [2, -2]:
        fig_res.add_hline(y=sigma_mult * res.std(), line_dash="dash",
                           line_color="#ff5252",
                           annotation_text=f"{sigma_mult:+d}σ")
    fig_res.update_layout(
        template="plotly_dark", height=280,
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        title="Residual Plot — Ensemble Model",
        xaxis_title="Fitted Value (kWh)", yaxis_title="Residual",
    )
    st.plotly_chart(fig_res, use_container_width=True)

    # Model metrics table
    st.markdown("#### 📋 Metrik Model")
    metrics_df = pd.DataFrame([{
        "Model": "VotingRegressor (Ridge + RF + HGB)",
        "R² Test": f"{model_r['r2_test']:.4f}",
        "MAE": f"{model_r['mae_test']:.4f} kWh",
        "RMSE": f"{model_r['rmse_test']:.4f} kWh",
        "Features": len(model_r["features"]),
        "Machine": machine_id,
    }])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 5 – STRESS TEST
# ══════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("### ⚡ Stress Test — Simulasi Kenaikan Produksi")
    st.markdown(
        "Proyeksi konsumsi energi ketika kapasitas produksi ditingkatkan. "
        "Termasuk efek **degradasi efisiensi** akibat overload."
    )

    st.dataframe(stress_df.style.background_gradient(
        subset=["Energi (kWh)"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True)

    fig_stress = make_subplots(rows=1, cols=2,
                                subplot_titles=["Energi vs Kenaikan Produksi",
                                                "Efisiensi Mesin vs Kenaikan"])
    scenarios = stress_df["Kenaikan Produksi (%)"].values
    energies  = stress_df["Energi (kWh)"].values
    effs      = stress_df["Efisiensi"].values

    clrs_stress = ["#69f0ae" if e == energies.min() else
                   "#ff5252" if e == energies.max() else "#4fc3f7"
                   for e in energies]

    fig_stress.add_trace(
        go.Bar(x=[f"+{s}%" for s in scenarios], y=energies,
               marker_color=clrs_stress, text=[f"{e:.2f}" for e in energies],
               textposition="outside", name="Energi"),
        row=1, col=1
    )
    fig_stress.add_trace(
        go.Scatter(x=[f"+{s}%" for s in scenarios], y=effs,
                   mode="lines+markers", name="Efisiensi",
                   line=dict(color="#ffb74d", width=2),
                   marker=dict(size=8)),
        row=1, col=2
    )

    fig_stress.update_layout(
        template="plotly_dark", height=380,
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        showlegend=False,
    )
    fig_stress.update_xaxes(tickfont=dict(color="#8b949e"),
                             gridcolor="#30363d")
    fig_stress.update_yaxes(tickfont=dict(color="#8b949e"),
                             gridcolor="#30363d")
    st.plotly_chart(fig_stress, use_container_width=True)

    # Custom stress scenario
    st.markdown("---")
    st.markdown("#### 🔧 Simulasi Skenario Kustom")
    col_cust1, col_cust2 = st.columns(2)
    with col_cust1:
        custom_inc = st.slider("Kenaikan Produksi (%)", 0, 150, 30, key="custom_inc")
        custom_eff = st.slider("Efisiensi Mesin", 0.85, 1.02, 0.95,
                                step=0.01, key="custom_eff", format="%.2f")
    with col_cust2:
        factor = 1 + custom_inc / 100
        custom_energy = predict_energy(model_r, {
            "X_Temp":         min(70 + custom_inc * 0.25, 110),
            "X_Load":         min(70 * factor, 100),
            "X_RPM":          min(1400 * factor, 2400),
            "X_Ambient_Temp": 30.0,
            "X_Humidity":     70.0,
            "X_Efficiency":   custom_eff,
        })
        base_e = predict_energy(model_r, {
            "X_Temp": 70, "X_Load": 70, "X_RPM": 1400,
            "X_Ambient_Temp": 30, "X_Humidity": 70, "X_Efficiency": 0.97,
        })
        delta_pct = (custom_energy - base_e) / base_e * 100
        extra_cost = (custom_energy - base_e) * annual_hours * tariff

        st.metric("Energi Diprediksi", f"{custom_energy:.3f} kWh",
                   delta=f"{delta_pct:+.1f}% vs baseline", delta_color="inverse")
        if extra_cost > 0:
            st.warning(f"⚠️ Biaya tambahan: Rp {extra_cost/1e6:.1f} Juta/tahun")
        else:
            st.success(f"✅ Hemat: Rp {-extra_cost/1e6:.1f} Juta/tahun")


# ══════════════════════════════════════════════════════════════════
# TAB 6 – ALERT MONITOR
# ══════════════════════════════════════════════════════════════════

with tab6:
    st.markdown("### 🚨 Real-time Alert Monitor")
    st.markdown(
        "Simulasi streaming sensor dengan 3 jenis alert: "
        "SPC Violation, EWMA Trend, dan Rate-of-Change."
    )

    col_al1, col_al2 = st.columns([1, 2])
    with col_al1:
        n_stream = st.slider("Jumlah pembacaan streaming", 100, 500, 300,
                              key="n_stream_slider")
        run_stream = st.button("▶️ Jalankan Simulasi Streaming", use_container_width=True)

    if run_stream or "alert_results" in st.session_state:
        if run_stream:
            with st.spinner("📡 Memproses stream data sensor…"):
                engine = simulate_streaming(df, spc_r, machine_id=machine_id,
                                             n_stream=n_stream)
                st.session_state["alert_results"] = engine.get_summary()
                st.session_state["alert_engine"]  = engine

        summary = st.session_state["alert_results"]

        with col_al2:
            al1, al2, al3 = st.columns(3)
            al1.metric("Total Diproses",  f"{summary['total_processed']}")
            al2.metric("Total Alert",     f"{summary['total_alerts']}",
                        delta_color="inverse")
            al3.metric("Alert Rate",
                        f"{summary['alert_rate']*100:.1f}%",
                        delta_color="inverse")

        st.markdown("---")
        st.markdown("#### 🗂️ Alert Log Terbaru")

        all_alerts = st.session_state["alert_engine"].alerts
        if all_alerts:
            # Kategorikan alert
            spc_alerts    = [a for a in all_alerts if "SPC VIOLATION" in a]
            ewma_alerts   = [a for a in all_alerts if "EWMA TREND"    in a]
            change_alerts = [a for a in all_alerts if "RAPID CHANGE"  in a]

            col_cat1, col_cat2, col_cat3 = st.columns(3)
            col_cat1.metric("🚨 SPC Violations",  len(spc_alerts))
            col_cat2.metric("📈 EWMA Trends",     len(ewma_alerts))
            col_cat3.metric("⚡ Rapid Changes",   len(change_alerts))

            st.markdown("**50 Alert Terbaru:**")
            for alert in reversed(all_alerts[-50:]):
                if "SPC VIOLATION" in alert:
                    st.markdown(f'<div class="alert-critical">{alert}</div>',
                                unsafe_allow_html=True)
                elif "EWMA" in alert:
                    st.markdown(f'<div class="alert-warning">{alert}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-info">{alert}</div>',
                                unsafe_allow_html=True)

            # Alert timeline chart
            if len(all_alerts) > 1:
                alert_types = []
                for a in all_alerts:
                    if "SPC VIOLATION" in a:   alert_types.append("SPC")
                    elif "EWMA TREND"  in a:   alert_types.append("EWMA")
                    else:                       alert_types.append("ROC")

                al_df = pd.DataFrame({
                    "Index": range(len(alert_types)),
                    "Type":  alert_types,
                })
                type_counts = al_df["Type"].value_counts().reset_index()
                type_counts.columns = ["Type", "Count"]
                fig_alert_pie = px.pie(
                    type_counts, values="Count", names="Type",
                    color_discrete_sequence=["#ff5252", "#ffb74d", "#4fc3f7"],
                    template="plotly_dark", height=280,
                    title="Distribusi Tipe Alert",
                )
                fig_alert_pie.update_layout(paper_bgcolor="#0d1117")
                st.plotly_chart(fig_alert_pie, use_container_width=True)
        else:
            st.info("✅ Tidak ada alert terdeteksi pada rentang streaming ini.")
    else:
        st.info("👆 Klik tombol **Jalankan Simulasi Streaming** untuk memulai.")


# ── Footer ──
st.divider()
st.markdown(
    "<p style='text-align:center; color:#444; font-size:12px;'>"
    "EcoSense v2.0 | AHM Manufacturing | "
    "VotingRegressor (Ridge+RF+HGB) · NSGA-II Pareto · CUSUM/EWMA SPC · 3-Detector Ensemble"
    "</p>",
    unsafe_allow_html=True,
)
