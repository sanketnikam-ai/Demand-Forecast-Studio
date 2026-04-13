import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from sklearn.linear_model import LinearRegression
import warnings
import io

warnings.filterwarnings("ignore")

# ── Page config ──
st.set_page_config(
    page_title="Demand Forecast Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric { background: #f8f9fb; border-radius: 10px; padding: 12px 16px; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .best-banner { background: #e6f1fb; border-left: 4px solid #378ADD; border-radius: 6px; padding: 12px 16px; margin-bottom: 1rem; }
    .param-box { background: #f0f7ee; border-left: 4px solid #27AE60; border-radius: 6px; padding: 14px 18px; margin: 12px 0; font-size: 0.88rem; }
    .param-box code { background: #e8f5e3; padding: 2px 6px; border-radius: 4px; font-family: Consolas, monospace; }
    section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Dimension columns ──
DIM_COLS = ["Key", "Channel", "Region", "ZSM", "Category", "Brand", "Sub Brand", "Track", "AOP-Track"]
UOM_LABELS = {"Val.": "Value (₹)", "Vol(Cs.)": "Volume (Cases)", "Vol(EA)": "Volume (Eaches)",
              "Vol(MT.)": "Volume (MT)", "Vol(ML.)": "Volume (ML)"}


# ══════════════════════════════════════════════════
# ── FORECASTING METHODS — all return (fitted, future, params_dict)
# ══════════════════════════════════════════════════

def run_sma(values, horizon, window=3):
    """Simple Moving Average."""
    fitted = []
    for i in range(len(values)):
        if i < window - 1:
            fitted.append(np.nan)
        else:
            fitted.append(np.mean(values[i - window + 1: i + 1]))
    recent = list(values[-window:])
    future = []
    for _ in range(horizon):
        fv = np.mean(recent)
        future.append(fv)
        recent.pop(0)
        recent.append(fv)
    params = {
        "Window size": window,
        "Last window values": ", ".join([f"{v:.2f}" for v in values[-window:]]),
        "Excel formula": f"=AVERAGE(last {window} cells)",
        "Recursive": "Yes — each forecast feeds back into window"
    }
    return np.array(fitted), np.array(future), params


def run_wma(values, horizon, window=3):
    """Weighted Moving Average."""
    weights = np.arange(1, window + 1, dtype=float)
    fitted = []
    for i in range(len(values)):
        if i < window - 1:
            fitted.append(np.nan)
        else:
            sl = values[i - window + 1: i + 1]
            fitted.append(np.dot(sl, weights) / weights.sum())
    recent = list(values[-window:])
    future = []
    for _ in range(horizon):
        fv = np.dot(recent, weights) / weights.sum()
        future.append(fv)
        recent.pop(0)
        recent.append(fv)
    params = {
        "Window size": window,
        "Weights": ", ".join([str(int(w)) for w in weights]),
        "Weight sum": int(weights.sum()),
        "Last window values": ", ".join([f"{v:.2f}" for v in values[-window:]]),
        "Excel formula": f"=(1*cell_t-2 + 2*cell_t-1 + 3*cell_t) / 6",
        "Recursive": "Yes — each forecast feeds back into window"
    }
    return np.array(fitted), np.array(future), params


def run_ses(values, horizon):
    """Single Exponential Smoothing."""
    model = SimpleExpSmoothing(
        values, initialization_method="estimated"
    ).fit(optimized=True)
    fitted = model.fittedvalues
    future = model.forecast(horizon)
    params = {
        "α (alpha)": round(model.params["smoothing_level"], 6),
        "Initial level L(0)": round(model.params["initial_level"], 4),
        "Final level": round(float(model.level[-1]), 4),
        "Excel formula": "F(t+1) = α × Y(t) + (1-α) × F(t)",
        "Excel init": f"C2 = {round(model.params['initial_level'], 4)}",
        "Excel recursion": f"C3 = {round(model.params['smoothing_level'], 6)}*B2 + {round(1-model.params['smoothing_level'], 6)}*C2",
        "Forecast note": "All future values are identical (no trend component)"
    }
    return np.array(fitted), np.array(future), params


def run_des(values, horizon):
    """Double Exponential Smoothing (Holt's Linear)."""
    model = ExponentialSmoothing(
        values, trend="add", seasonal=None,
        initialization_method="estimated"
    ).fit(optimized=True)
    fitted = model.fittedvalues
    future = model.forecast(horizon)
    params = {
        "α (alpha — level)": round(model.params["smoothing_level"], 6),
        "β (beta — trend)": round(model.params["smoothing_trend"], 6),
        "Initial level L(0)": round(model.params["initial_level"], 4),
        "Initial trend T(0)": round(model.params["initial_trend"], 4),
        "Final level L(T)": round(float(model.level[-1]), 4),
        "Final trend T(T)": round(float(model.trend[-1]), 4),
        "Excel — Level formula": f"L(t) = {round(model.params['smoothing_level'],6)} × Y(t) + {round(1-model.params['smoothing_level'],6)} × (L(t-1) + T(t-1))",
        "Excel — Trend formula": f"T(t) = {round(model.params['smoothing_trend'],6)} × (L(t) - L(t-1)) + {round(1-model.params['smoothing_trend'],6)} × T(t-1)",
        "Excel — Forecast": "F(t+h) = L(T) + h × T(T)",
        "Excel — Init level cell": f"C2 = {round(model.params['initial_level'], 4)}",
        "Excel — Init trend cell": f"D2 = {round(model.params['initial_trend'], 4)}",
    }
    return np.array(fitted), np.array(future), params


def run_holt_winters(values, horizon, seasonal_periods=12):
    """Holt-Winters (Triple Exponential Smoothing)."""
    n = len(values)
    if n < 2 * seasonal_periods:
        seasonal_periods = max(2, n // 3)
    seasonal_type = "mul"
    try:
        model = ExponentialSmoothing(
            values, trend="add", seasonal="mul",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        ).fit(optimized=True)
    except Exception:
        try:
            seasonal_type = "add"
            model = ExponentialSmoothing(
                values, trend="add", seasonal="add",
                seasonal_periods=seasonal_periods,
                initialization_method="estimated"
            ).fit(optimized=True)
        except Exception:
            fitted, future, des_params = run_des(values, horizon)
            des_params["Fallback"] = "Holt-Winters failed, fell back to DES"
            return fitted, future, des_params

    # Extract initial seasonal factors
    init_seasons = model.params.get("initial_seasons", None)
    season_str = ""
    if init_seasons is not None:
        season_vals = [round(float(s), 4) for s in init_seasons]
        season_str = ", ".join([str(s) for s in season_vals])

    params = {
        "α (alpha — level)": round(model.params["smoothing_level"], 6),
        "β (beta — trend)": round(model.params["smoothing_trend"], 6),
        "γ (gamma — seasonal)": round(model.params["smoothing_seasonal"], 6),
        "Seasonal type": "Multiplicative" if seasonal_type == "mul" else "Additive",
        "Seasonal period (m)": seasonal_periods,
        "Initial level L(0)": round(model.params["initial_level"], 4),
        "Initial trend T(0)": round(model.params["initial_trend"], 4),
        "Initial seasonal factors S(1)..S(m)": season_str if season_str else "N/A",
        "Final level L(T)": round(float(model.level[-1]), 4),
        "Final trend T(T)": round(float(model.trend[-1]), 4),
    }
    if seasonal_type == "mul":
        params["Excel — Level"] = f"L(t) = α×(Y(t)/S(t-m)) + (1-α)×(L(t-1)+T(t-1))"
        params["Excel — Seasonal"] = f"S(t) = γ×(Y(t)/L(t)) + (1-γ)×S(t-m)"
        params["Excel — Forecast"] = "F(t+h) = (L(T) + h×T(T)) × S(t+h-m)"
    else:
        params["Excel — Level"] = f"L(t) = α×(Y(t)-S(t-m)) + (1-α)×(L(t-1)+T(t-1))"
        params["Excel — Seasonal"] = f"S(t) = γ×(Y(t)-L(t)) + (1-γ)×S(t-m)"
        params["Excel — Forecast"] = "F(t+h) = (L(T) + h×T(T)) + S(t+h-m)"
    params["Excel — Trend"] = "T(t) = β×(L(t)-L(t-1)) + (1-β)×T(t-1)"

    return np.array(model.fittedvalues), np.array(model.forecast(horizon)), params


def run_linear_seasonal(values, horizon, seasonal_periods=12):
    """Linear Regression + Seasonal Dummies."""
    n = len(values)
    t = np.arange(n).reshape(-1, 1)
    months = np.array([i % seasonal_periods for i in range(n)])
    dummies = np.zeros((n, seasonal_periods - 1))
    for i in range(n):
        m = months[i]
        if m < seasonal_periods - 1:
            dummies[i, m] = 1.0
    X = np.hstack([t, dummies])
    model = LinearRegression().fit(X, values)
    fitted = model.predict(X)
    # Future
    t_f = np.arange(n, n + horizon).reshape(-1, 1)
    months_f = np.array([i % seasonal_periods for i in range(n, n + horizon)])
    dummies_f = np.zeros((horizon, seasonal_periods - 1))
    for i in range(horizon):
        m = months_f[i]
        if m < seasonal_periods - 1:
            dummies_f[i, m] = 1.0
    X_f = np.hstack([t_f, dummies_f])
    future = model.predict(X_f)

    # Build params — coefficients with month labels
    month_labels = ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
    params = {
        "Intercept (β0 — April baseline)": round(model.intercept_, 4),
        "Trend (β1 — per month)": round(model.coef_[0], 4),
        "R² score": round(model.score(X, values), 4),
    }
    for i, lbl in enumerate(month_labels):
        coef = round(model.coef_[i + 1], 4)
        sign = "+" if coef >= 0 else ""
        params[f"Seasonal {lbl} (vs April)"] = f"{sign}{coef}",
    params["Excel formula"] = f"Y(t) = {round(model.intercept_,2)} + {round(model.coef_[0],4)}×t + seasonal_dummy_coeff"
    params["Excel tool"] = "Data tab → Data Analysis → Regression, or =LINEST()"

    return np.array(fitted), np.array(future), params


# Method registry
METHODS = {
    "holt_winters": ("Holt-Winters", run_holt_winters),
    "des": ("Double Exp. Smoothing", run_des),
    "ses": ("Single Exp. Smoothing", run_ses),
    "linear_seasonal": ("Linear Reg. + Seasonal", run_linear_seasonal),
    "wma": ("Weighted Moving Avg", run_wma),
    "sma": ("Simple Moving Avg", run_sma),
}


# ══════════════════════════════════════════════════
# ── ACCURACY METRICS
# ══════════════════════════════════════════════════

def calc_metrics(actual, fitted):
    mask = ~np.isnan(fitted) & ~np.isnan(actual)
    a, f = actual[mask], fitted[mask]
    if len(a) == 0:
        return 999, 999, 999, 999
    sum_actual = np.sum(np.abs(a))
    wmape = (np.sum(np.abs(a - f)) / sum_actual * 100) if sum_actual > 0 else 999
    bias_pct = (np.sum(f - a) / sum_actual * 100) if sum_actual > 0 else 0
    mae = np.mean(np.abs(a - f))
    rmse = np.sqrt(np.mean((a - f) ** 2))
    return round(wmape, 2), round(bias_pct, 2), round(mae, 2), round(rmse, 2)


def pick_best_method(valid_results, bias_threshold=5.0):
    low_bias = {k: v for k, v in valid_results.items() if abs(v["bias"]) <= bias_threshold}
    pool = low_bias if low_bias else valid_results
    return min(pool, key=lambda k: pool[k]["wmape"])


# ══════════════════════════════════════════════════
# ── SIDEBAR
# ══════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📊 Demand Forecast Studio")
    st.caption("Upload data · Filter · Forecast")
    st.divider()

    st.markdown("##### ① Upload master data")
    uploaded = st.file_uploader("Upload .xlsx file", type=["xlsx"], label_visibility="collapsed")

    if uploaded:
        @st.cache_data(show_spinner="Loading data...")
        def load_data(file_bytes):
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
            df = df[df["Key"].notna() & (df["Key"] != "zzz")].reset_index(drop=True)
            return df

        raw = load_data(uploaded.getvalue())
        month_cols = [c for c in raw.columns if c not in DIM_COLS]

        st.success(f"✓ {len(raw):,} rows · {len(month_cols)} months loaded")
        st.divider()

        st.markdown("##### ② Unit of measure")
        available_uoms = sorted(raw["Key"].unique())
        uom_options = {UOM_LABELS.get(k, k): k for k in available_uoms}
        selected_uom_label = st.selectbox("UOM", options=list(uom_options.keys()), label_visibility="collapsed")
        selected_uom = uom_options[selected_uom_label]
        df_filtered = raw[raw["Key"] == selected_uom].copy()
        st.divider()

        st.markdown("##### ③ Dimension filters")
        filter_cols = ["Channel", "Region", "ZSM", "Category", "Brand", "Sub Brand", "Track", "AOP-Track"]
        selections = {}
        for col in filter_cols:
            opts = sorted(df_filtered[col].dropna().unique())
            choice = st.selectbox(col, ["All"] + list(opts), key=f"filter_{col}")
            if choice != "All":
                df_filtered = df_filtered[df_filtered[col] == choice]
            selections[col] = choice
        st.divider()

        st.markdown("##### ④ Forecast settings")
        horizon = st.slider("Forecast horizon (months)", 1, 24, 6)
        run_btn = st.button("🚀 Run forecast", use_container_width=True, type="primary")

    else:
        run_btn = False


# ══════════════════════════════════════════════════
# ── MAIN AREA
# ══════════════════════════════════════════════════

if not uploaded:
    st.markdown("## 📊 Demand Forecast Studio")
    st.info("Upload your master data (.xlsx) in the sidebar to get started.", icon="⬆️")
    st.markdown("""
    **How it works:**
    1. Upload your Base_Data.xlsx file
    2. Select unit of measure (Value, Volume, etc.)
    3. Filter by any dimension combination
    4. Click **Run Forecast** to compare 6 statistical methods
    5. View results, compare accuracy, and export forecasts

    **Methods (all Excel-verifiable):**
    Holt-Winters · Double Exp. Smoothing · Single Exp. Smoothing ·
    Linear Reg. + Seasonal · Weighted Moving Avg · Simple Moving Avg

    **Accuracy metrics:** WMAPE (primary), Bias %, MAE, RMSE
    """)
    st.stop()

if not run_btn and "results" not in st.session_state:
    st.markdown("## 📊 Demand Forecast Studio")
    st.info("Configure your filters in the sidebar and click **Run Forecast**.", icon="⚙️")
    st.markdown(f"**Filtered rows:** {len(df_filtered):,} | **UOM:** {selected_uom_label}")
    active_filters = [f"{k}: {v}" for k, v in selections.items() if v != "All"]
    if active_filters:
        st.markdown("**Active filters:** " + " → ".join(active_filters))
    st.stop()


# ══════════════════════════════════════════════════
# ── RUN FORECASTS
# ══════════════════════════════════════════════════

if run_btn:
    agg = df_filtered[month_cols].sum()
    values = agg.values.astype(float)

    first_nonzero = 0
    for i, v in enumerate(values):
        if v > 0:
            first_nonzero = i
            break
    values = values[first_nonzero:]
    labels = [str(c) for c in month_cols[first_nonzero:]]

    if len(values) < 6:
        st.error("Not enough data points after filtering. Need at least 6 months of non-zero data.")
        st.stop()

    results = {}
    for key, (name, func) in METHODS.items():
        try:
            fitted, future, params = func(values, horizon)
            wmape, bias, mae, rmse = calc_metrics(values, fitted)
            results[key] = {"name": name, "fitted": fitted, "future": future,
                            "wmape": wmape, "bias": bias, "mae": mae, "rmse": rmse,
                            "params": params}
        except Exception:
            results[key] = None

    st.session_state["results"] = results
    st.session_state["values"] = values
    st.session_state["labels"] = labels
    st.session_state["horizon"] = horizon
    st.session_state["selections"] = selections.copy()
    st.session_state["uom_label"] = selected_uom_label


# ══════════════════════════════════════════════════
# ── DISPLAY RESULTS
# ══════════════════════════════════════════════════

if "results" in st.session_state:
    results = st.session_state["results"]
    values = st.session_state["values"]
    labels = st.session_state["labels"]
    horizon = st.session_state["horizon"]
    sel = st.session_state["selections"]
    uom_label = st.session_state["uom_label"]

    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        st.error("All forecasting methods failed. Try a different filter combination with more data.")
        st.stop()

    best_key = pick_best_method(valid)
    best = valid[best_key]

    # Title
    active = [f"{k}: {v}" for k, v in sel.items() if v != "All"]
    title_suffix = " → ".join(active) if active else "All dimensions"
    st.markdown(f"## Forecast results — {uom_label}")
    st.caption(title_suffix)

    # Best method banner
    bias_note = ""
    if abs(best["bias"]) > 5:
        bias_note = f' (⚠️ Bias {best["bias"]:+.1f}% exceeds ±5%)'
    st.markdown(f"""<div class="best-banner">
        💡 <strong>Recommended:</strong> <span style="color:#378ADD">{best['name']}</span>
        — lowest WMAPE at <strong>{best['wmape']}%</strong> among methods with acceptable bias{bias_note}
    </div>""", unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Best method", best["name"])
    m2.metric("WMAPE", f"{best['wmape']}%")
    bias_dir = "Over ↑" if best["bias"] > 0 else "Under ↓" if best["bias"] < 0 else "Neutral"
    m3.metric("Bias", f"{best['bias']:+.1f}%", delta=bias_dir,
              delta_color="inverse" if best["bias"] > 0 else "normal")
    m4.metric("Data points", f"{len(values)}")
    m5.metric("Next forecast", f"{best['future'][0]:,.1f}")

    st.markdown("---")

    # ── Method selector ──
    method_options = {v["name"]: k for k, v in valid.items()}
    selected_method_name = st.selectbox("Select method to visualize",
                                         list(method_options.keys()),
                                         index=list(method_options.keys()).index(best["name"]))
    sel_key = method_options[selected_method_name]
    sel_result = valid[sel_key]

    # Generate future labels
    last_label = labels[-1]
    month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                 "JUL": 7, "AUG": 8, "SEPT": 9, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

    def parse_month_label(lbl):
        parts = lbl.strip().split()
        m = month_map.get(parts[0].upper(), 1)
        y = int(parts[1]) if len(parts) > 1 else 2026
        return m, y

    lm, ly = parse_month_label(last_label)
    future_labels = []
    month_names = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEPT", "OCT", "NOV", "DEC"]
    cm, cy = lm, ly
    for _ in range(horizon):
        cm += 1
        if cm > 12:
            cm = 1
            cy += 1
        future_labels.append(f"{month_names[cm - 1]} {cy}")

    # ── Parameters panel — linked to selected method ──
    sel_params = sel_result.get("params", {})
    if sel_params:
        param_lines = []
        for pk, pv in sel_params.items():
            val = pv[0] if isinstance(pv, tuple) else pv
            param_lines.append(f"<code>{pk}</code> = {val}")
        params_html = "<br>".join(param_lines)
        st.markdown(f"""<div class="param-box">
            <strong>🔧 Parameters — {selected_method_name}</strong>
            <br><span style="font-size:0.8rem; color:#555;">Use these exact values in Excel to replicate the tool's output</span>
            <br><br>{params_html}
        </div>""", unsafe_allow_html=True)

    # ── Chart ──
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=values, mode="lines+markers", name="Actual",
        line=dict(color="#378ADD", width=2.5), marker=dict(size=3),
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=sel_result["fitted"], mode="lines", name="Fitted",
        line=dict(color="#1D9E75", width=1.8, dash="dash"),
    ))
    fc_x = [labels[-1]] + future_labels
    fc_y = [values[-1]] + list(sel_result["future"])
    fig.add_trace(go.Scatter(
        x=fc_x, y=fc_y, mode="lines+markers", name="Forecast",
        line=dict(color="#7F77DD", width=2.5), marker=dict(size=6),
    ))
    fig.add_vrect(x0=labels[-1], x1=future_labels[-1],
                  fillcolor="rgba(127,119,221,0.08)", line_width=0,
                  annotation_text="Forecast zone", annotation_position="top left",
                  annotation_font_size=10, annotation_font_color="#7F77DD")
    fig.update_layout(
        title=f"Actual vs Fitted vs Forecast — {selected_method_name}",
        xaxis_title="Period", yaxis_title=uom_label,
        height=420, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=80),
        xaxis=dict(tickangle=-45, dtick=3),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Method comparison table ──
    st.markdown("### Method comparison")
    st.caption("Ranked by WMAPE. ★ = recommended (lowest WMAPE among methods with |Bias| ≤ 5%)")
    comp_data = []
    for k, v in valid.items():
        bias_flag = "⚠️" if abs(v["bias"]) > 5 else "✓"
        comp_data.append({
            "Method": v["name"] + (" ★" if k == best_key else ""),
            "WMAPE %": v["wmape"],
            "Bias %": v["bias"],
            "Bias Check": bias_flag,
            "MAE": v["mae"],
            "RMSE": v["rmse"],
            "Next Forecast": round(v["future"][0], 2),
        })
    comp_df = pd.DataFrame(comp_data).sort_values("WMAPE %").reset_index(drop=True)

    def highlight_best(row):
        if "★" in str(row["Method"]):
            return ["background-color: #e6f1fb"] * len(row)
        return [""] * len(row)

    st.dataframe(
        comp_df.style.apply(highlight_best, axis=1).format({
            "WMAPE %": "{:.1f}", "Bias %": "{:+.1f}", "MAE": "{:.1f}",
            "RMSE": "{:.1f}", "Next Forecast": "{:,.1f}"
        }),
        use_container_width=True, hide_index=True
    )

    # ── Forecast output — linked to selected method ──
    st.markdown(f"### Forecast output — {selected_method_name}")
    if sel_key != best_key:
        st.caption(f"Showing forecast for selected method. Recommended method is {best['name']}.")

    for row_start in range(0, horizon, 6):
        row_end = min(row_start + 6, horizon)
        fc_cols = st.columns(row_end - row_start)
        for i, col in enumerate(fc_cols):
            idx = row_start + i
            with col:
                st.metric(future_labels[idx], f"{sel_result['future'][idx]:,.1f}")

    st.markdown(f"""
    <div style="background: #f8f9fb; border-radius: 8px; padding: 10px 16px; margin-top: 8px; font-size: 0.85rem;">
        <strong>{selected_method_name}</strong> —
        WMAPE: {sel_result['wmape']}% ·
        Bias: {sel_result['bias']:+.1f}% ·
        MAE: {sel_result['mae']:.1f} ·
        RMSE: {sel_result['rmse']:.1f}
    </div>
    """, unsafe_allow_html=True)

    # ── Export ──
    st.markdown("---")
    export_rows = []
    for i, lbl in enumerate(future_labels):
        row = {"Period": lbl}
        for k, v in valid.items():
            row[v["name"]] = round(v["future"][i], 2)
        export_rows.append(row)
    export_df = pd.DataFrame(export_rows)

    hist_rows = []
    for i, lbl in enumerate(labels):
        row = {"Period": lbl, "Actual": round(values[i], 2)}
        for k, v in valid.items():
            f_val = v["fitted"][i]
            row[v["name"] + " (Fitted)"] = round(f_val, 2) if not np.isnan(f_val) else ""
        hist_rows.append(row)
    hist_df = pd.DataFrame(hist_rows)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        hist_df.to_excel(writer, sheet_name="Historical + Fitted", index=False)
        export_df.to_excel(writer, sheet_name="Forecast", index=False)
        # Metrics sheet
        metrics_rows = []
        for k, v in valid.items():
            metrics_rows.append({"Method": v["name"], "WMAPE %": v["wmape"],
                                  "Bias %": v["bias"], "MAE": v["mae"], "RMSE": v["rmse"]})
        pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="Accuracy Metrics", index=False)
        # Parameters sheet — all methods, all params
        param_rows = []
        for k, v in valid.items():
            p = v.get("params", {})
            for pk, pv in p.items():
                val = pv[0] if isinstance(pv, tuple) else pv
                param_rows.append({"Method": v["name"], "Parameter": pk, "Value": str(val)})
        pd.DataFrame(param_rows).to_excel(writer, sheet_name="Model Parameters", index=False)

    st.download_button(
        "📥 Download full results (Excel)",
        data=buffer.getvalue(),
        file_name="demand_forecast_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
