import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Dimension columns ──
DIM_COLS = ["Key", "Channel", "Region", "ZSM", "Category", "Brand", "Sub Brand", "Track", "AOP-Track"]
UOM_LABELS = {"Val.": "Value (₹)", "Vol(Cs.)": "Volume (Cases)", "Vol(EA)": "Volume (Eaches)",
              "Vol(MT.)": "Volume (MT)", "Vol(ML.)": "Volume (ML)"}


# ── Forecasting engines ──
def run_wma(values, horizon, window=3):
    weights = np.arange(1, window + 1, dtype=float)
    fitted = []
    for i in range(len(values)):
        if i < window - 1:
            fitted.append(np.nan)
        else:
            sl = values[i - window + 1: i + 1]
            fitted.append(np.dot(sl, weights) / weights.sum())
    last = values[-window:]
    fv = np.dot(last, weights) / weights.sum()
    future = [fv] * horizon
    return np.array(fitted), np.array(future)


def run_ses(values, horizon):
    model = SimpleExpSmoothing(values, initialization_method="estimated").fit(optimized=True)
    fitted = model.fittedvalues
    future = model.forecast(horizon)
    return np.array(fitted), np.array(future)


def run_holt_winters(values, horizon, seasonal_periods=12):
    n = len(values)
    if n < 2 * seasonal_periods:
        seasonal_periods = max(2, n // 3)
    try:
        model = ExponentialSmoothing(
            values, trend="add", seasonal="mul",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        ).fit(optimized=True)
    except Exception:
        try:
            model = ExponentialSmoothing(
                values, trend="add", seasonal="add",
                seasonal_periods=seasonal_periods,
                initialization_method="estimated"
            ).fit(optimized=True)
        except Exception:
            return run_ses(values, horizon)
    return np.array(model.fittedvalues), np.array(model.forecast(horizon))


def run_sarima(values, horizon, seasonal_periods=12):
    n = len(values)
    if n < 2 * seasonal_periods:
        seasonal_periods = max(2, n // 3)
    try:
        model = SARIMAX(values, order=(1, 1, 1),
                        seasonal_order=(1, 0, 1, seasonal_periods),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False, maxiter=100)
        fitted = model.fittedvalues
        future = model.forecast(horizon)
        fitted = np.where(np.isnan(fitted), values, fitted)
    except Exception:
        return run_ses(values, horizon)
    return np.array(fitted), np.array(future)


def run_linear_seasonal(values, horizon, seasonal_periods=12):
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
    return np.array(fitted), np.array(future)


METHODS = {
    "holt_winters": ("Holt-Winters", run_holt_winters),
    "sarima": ("SARIMA", run_sarima),
    "ses": ("Simple Exp. Smoothing", run_ses),
    "linear_seasonal": ("Linear Reg. + Seasonal", run_linear_seasonal),
    "wma": ("Weighted Moving Avg", run_wma),
}


def calc_metrics(actual, fitted):
    mask = ~np.isnan(fitted) & ~np.isnan(actual)
    a, f = actual[mask], fitted[mask]
    if len(a) == 0:
        return 999, 999, 999
    mae = np.mean(np.abs(a - f))
    mape = np.mean(np.where(a != 0, np.abs((a - f) / a), 0)) * 100
    rmse = np.sqrt(np.mean((a - f) ** 2))
    return round(mape, 2), round(mae, 2), round(rmse, 2)


# ── Sidebar ──
with st.sidebar:
    st.markdown("### 📊 Demand Forecast Studio")
    st.caption("Upload data · Filter · Forecast")
    st.divider()

    # 1. File upload
    st.markdown("##### ① Upload master data")
    uploaded = st.file_uploader("Upload .xlsx file", type=["xlsx"], label_visibility="collapsed")

    if uploaded:
        @st.cache_data(show_spinner="Loading data...")
        def load_data(file_bytes):
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
            # Clean: drop rows where Key is null or 'zzz'
            df = df[df["Key"].notna() & (df["Key"] != "zzz")].reset_index(drop=True)
            return df

        raw = load_data(uploaded.getvalue())
        month_cols = [c for c in raw.columns if c not in DIM_COLS]

        st.success(f"✓ {len(raw):,} rows · {len(month_cols)} months loaded")
        st.divider()

        # 2. UOM
        st.markdown("##### ② Unit of measure")
        available_uoms = sorted(raw["Key"].unique())
        uom_options = {UOM_LABELS.get(k, k): k for k in available_uoms}
        selected_uom_label = st.selectbox("UOM", options=list(uom_options.keys()), label_visibility="collapsed")
        selected_uom = uom_options[selected_uom_label]
        df_filtered = raw[raw["Key"] == selected_uom].copy()
        st.divider()

        # 3. Cascading filters
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

        # 4. Forecast settings
        st.markdown("##### ④ Forecast settings")
        horizon = st.slider("Forecast horizon (months)", 1, 24, 6)
        run_btn = st.button("🚀 Run forecast", use_container_width=True, type="primary")

    else:
        run_btn = False

# ── Main area ──
if not uploaded:
    st.markdown("## 📊 Demand Forecast Studio")
    st.info("Upload your master data (.xlsx) in the sidebar to get started.", icon="⬆️")
    st.markdown("""
    **How it works:**
    1. Upload your Base_Data.xlsx file
    2. Select unit of measure (Value, Volume, etc.)
    3. Filter by any dimension combination
    4. Click **Run Forecast** to compare 5 statistical methods
    5. View results, compare accuracy, and export forecasts

    **Methods included:** Holt-Winters · SARIMA · Simple Exponential Smoothing · Linear Regression with Seasonal Dummies · Weighted Moving Average
    """)
    st.stop()

if not run_btn and "results" not in st.session_state:
    st.markdown("## 📊 Demand Forecast Studio")
    st.info("Configure your filters in the sidebar and click **Run Forecast**.", icon="⚙️")
    # Show data summary
    st.markdown(f"**Filtered rows:** {len(df_filtered):,} | **UOM:** {selected_uom_label}")
    active_filters = [f"{k}: {v}" for k, v in selections.items() if v != "All"]
    if active_filters:
        st.markdown("**Active filters:** " + " → ".join(active_filters))
    st.stop()

# ── Run forecasts ──
if run_btn:
    # Aggregate filtered data
    agg = df_filtered[month_cols].sum()
    values = agg.values.astype(float)

    # Clean: replace 0s at start
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
            fitted, future = func(values, horizon)
            mape, mae, rmse = calc_metrics(values, fitted)
            results[key] = {"name": name, "fitted": fitted, "future": future,
                            "mape": mape, "mae": mae, "rmse": rmse}
        except Exception as e:
            results[key] = None

    st.session_state["results"] = results
    st.session_state["values"] = values
    st.session_state["labels"] = labels
    st.session_state["horizon"] = horizon
    st.session_state["selections"] = selections.copy()
    st.session_state["uom_label"] = selected_uom_label
    st.session_state["df_filtered_count"] = len(df_filtered)

# ── Display results ──
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

    best_key = min(valid, key=lambda k: valid[k]["mape"])
    best = valid[best_key]

    # Title
    active = [f"{k}: {v}" for k, v in sel.items() if v != "All"]
    title_suffix = " → ".join(active) if active else "All dimensions"
    st.markdown(f"## Forecast results — {uom_label}")
    st.caption(title_suffix)

    # Best method banner
    st.markdown(f"""<div class="best-banner">
        💡 <strong>Recommended method:</strong> <span style="color:#378ADD">{best['name']}</span>
        — lowest MAPE at <strong>{best['mape']}%</strong> on your data
    </div>""", unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best method", best["name"])
    m2.metric("MAPE", f"{best['mape']}%")
    m3.metric("Data points", f"{len(values)}")
    m4.metric("Next forecast", f"{best['future'][0]:,.1f}")

    # ── Chart ──
    st.markdown("---")

    # Method selector for chart
    method_options = {v["name"]: k for k, v in valid.items()}
    selected_method_name = st.selectbox("Select method to visualize",
                                         list(method_options.keys()),
                                         index=list(method_options.keys()).index(best["name"]))
    sel_key = method_options[selected_method_name]
    sel_result = valid[sel_key]

    # Generate future labels
    last_label = labels[-1]
    # Parse last month
    import re
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

    all_labels = list(labels) + future_labels

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=labels, y=values, mode="lines+markers", name="Actual",
        line=dict(color="#378ADD", width=2.5), marker=dict(size=3),
    ))

    # Fitted
    fig.add_trace(go.Scatter(
        x=labels, y=sel_result["fitted"], mode="lines", name="Fitted",
        line=dict(color="#1D9E75", width=1.8, dash="dash"),
    ))

    # Forecast
    # Connect forecast to last actual
    fc_x = [labels[-1]] + future_labels
    fc_y = [values[-1]] + list(sel_result["future"])
    fig.add_trace(go.Scatter(
        x=fc_x, y=fc_y, mode="lines+markers", name="Forecast",
        line=dict(color="#7F77DD", width=2.5), marker=dict(size=6),
    ))

    # Forecast shaded region
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
    comp_data = []
    for k, v in valid.items():
        comp_data.append({
            "Method": v["name"] + (" ★" if k == best_key else ""),
            "MAPE %": v["mape"],
            "MAE": v["mae"],
            "RMSE": v["rmse"],
            "Next Forecast": round(v["future"][0], 2),
        })
    comp_df = pd.DataFrame(comp_data).sort_values("MAPE %").reset_index(drop=True)

    def highlight_best(row):
        if "★" in str(row["Method"]):
            return ["background-color: #e6f1fb"] * len(row)
        return [""] * len(row)

    st.dataframe(
        comp_df.style.apply(highlight_best, axis=1).format({
            "MAPE %": "{:.1f}", "MAE": "{:.1f}", "RMSE": "{:.1f}", "Next Forecast": "{:,.1f}"
        }),
        use_container_width=True, hide_index=True
    )

    # ── Forecast output ──
    st.markdown("### Forecast output")
    fc_cols = st.columns(min(horizon, 6))
    for i in range(min(horizon, 6)):
        with fc_cols[i]:
            st.metric(future_labels[i], f"{best['future'][i]:,.1f}")

    if horizon > 6:
        fc_cols2 = st.columns(min(horizon - 6, 6))
        for i in range(min(horizon - 6, 6)):
            with fc_cols2[i]:
                st.metric(future_labels[i + 6], f"{best['future'][i + 6]:,.1f}")

    # ── Export ──
    st.markdown("---")
    export_rows = []
    for i, lbl in enumerate(future_labels):
        row = {"Period": lbl}
        for k, v in valid.items():
            row[v["name"]] = round(v["future"][i], 2)
        export_rows.append(row)
    export_df = pd.DataFrame(export_rows)

    # Also add actuals + fitted for full picture
    hist_rows = []
    for i, lbl in enumerate(labels):
        row = {"Period": lbl, "Actual": round(values[i], 2)}
        for k, v in valid.items():
            f_val = v["fitted"][i]
            row[v["name"] + " (Fitted)"] = round(f_val, 2) if not np.isnan(f_val) else ""
        hist_rows.append(row)
    hist_df = pd.DataFrame(hist_rows)

    # Excel export
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        hist_df.to_excel(writer, sheet_name="Historical + Fitted", index=False)
        export_df.to_excel(writer, sheet_name="Forecast", index=False)
        # Metrics sheet
        metrics_rows = []
        for k, v in valid.items():
            metrics_rows.append({"Method": v["name"], "MAPE %": v["mape"],
                                  "MAE": v["mae"], "RMSE": v["rmse"]})
        pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="Accuracy Metrics", index=False)

    st.download_button(
        "📥 Download full results (Excel)",
        data=buffer.getvalue(),
        file_name="demand_forecast_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
