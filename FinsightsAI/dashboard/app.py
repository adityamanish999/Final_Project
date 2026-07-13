"""
app.py — AI-Powered Financial Market Prediction Dashboard
Streamlit multi-page dashboard for stock prediction, sentiment analysis,
and SHAP explainability.
"""

import sys
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings("ignore")

from src.utils import MODEL_DIR, PROC_DIR, TICKERS
from src.feature_engineer import ALL_FEATURES, TECHNICAL_FEATURES, SENTIMENT_FEATURES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI — Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark glassmorphism sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0f23 0%, #1a1a3e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Main background */
.main .block-container {
    background: #0d0d1a;
    padding-top: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(79,142,247,0.15) 0%, rgba(130,80,255,0.10) 100%);
    border: 1px solid rgba(79,142,247,0.3);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    backdrop-filter: blur(12px);
    margin: 0.3rem 0;
}
.metric-card h2 { color: #ffffff; margin: 0; font-size: 2rem; font-weight: 700; }
.metric-card p  { color: rgba(255,255,255,0.55); margin: 0.2rem 0 0; font-size: 0.85rem; }

/* Prediction badge */
.pred-up {
    background: linear-gradient(135deg, #00c88c, #00a870);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    color: white;
    letter-spacing: 1px;
    box-shadow: 0 0 40px rgba(0,200,140,0.4);
    animation: pulse-green 2s infinite;
}
.pred-down {
    background: linear-gradient(135deg, #ff4d6d, #c9184a);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    color: white;
    letter-spacing: 1px;
    box-shadow: 0 0 40px rgba(255,77,109,0.4);
    animation: pulse-red 2s infinite;
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 40px rgba(0,200,140,0.4); }
    50%      { box-shadow: 0 0 60px rgba(0,200,140,0.7); }
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 40px rgba(255,77,109,0.4); }
    50%      { box-shadow: 0 0 60px rgba(255,77,109,0.7); }
}

/* Section headings */
.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #ffffff;
    margin: 1.5rem 0 0.8rem;
    border-left: 4px solid #4f8ef7;
    padding-left: 0.8rem;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* Sidebar text */
section[data-testid="stSidebar"] * { color: #e0e0f0 !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] { color: #a0a0c0; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #4f8ef7 !important; border-bottom-color: #4f8ef7 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_feature_data(ticker: str) -> pd.DataFrame | None:
    path = PROC_DIR / f"{ticker}_features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    return df

@st.cache_resource(show_spinner=False)
def load_model(ticker: str, model_name: str):
    safe = model_name.replace(" ", "_").lower()
    path = MODEL_DIR / f"{ticker}_{safe}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)

def _plotly_dark_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff", size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#c0c0d0", family="Inter"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(0,0,0,0)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c0c0d0")),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 FinSight AI")
    st.markdown("*AI-Powered Market Prediction*")
    st.markdown("---")

    selected_ticker = st.selectbox("🏷️ Select Stock", TICKERS, index=0)
    selected_model  = st.selectbox("🤖 Select Model", MODEL_NAMES, index=2)   # XGBoost default

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Market Overview", "💬 Sentiment Trends",
         "🔮 Prediction Engine", "🔍 Explainability", "📉 Model Performance"],
        index=0,
    )
    st.markdown("---")
    st.caption("Data: Yahoo Finance + Kaggle Financial News")
    st.caption("Models: LR | RF | XGBoost | LightGBM")


# ── Load data ─────────────────────────────────────────────────────────────────

df = load_feature_data(selected_ticker)
model = load_model(selected_ticker, selected_model)

if df is None:
    st.error(
        f"⚠️ No feature data found for **{selected_ticker}**.\n\n"
        "Please run the preprocessing and model training notebooks first:\n"
        "`notebooks/01_data_preprocessing.ipynb` → `04_model_training.ipynb`"
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Market Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Market Overview":
    st.markdown(f"<div class='section-title'>📊 Market Overview — {selected_ticker}</div>", unsafe_allow_html=True)

    # KPI row
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    price_change = latest["Close"] - prev["Close"]
    pct_change   = (price_change / prev["Close"]) * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <h2>${latest['Close']:.2f}</h2>
            <p>Close Price</p></div>""", unsafe_allow_html=True)
    with c2:
        color = "🟢" if price_change >= 0 else "🔴"
        st.markdown(f"""<div class="metric-card">
            <h2>{color} {pct_change:+.2f}%</h2>
            <p>Daily Return</p></div>""", unsafe_allow_html=True)
    with c3:
        vol_m = latest["Volume"] / 1_000_000
        st.markdown(f"""<div class="metric-card">
            <h2>{vol_m:.1f}M</h2>
            <p>Volume</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <h2>{latest['rsi']:.1f}</h2>
            <p>RSI (14)</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Date range filter
    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    d_start, d_end = st.slider(
        "Date Range", min_value=min_d, max_value=max_d,
        value=(pd.Timestamp("2023-01-01").date(), max_d),
        format="YYYY-MM-DD",
    )
    view = df[(df["Date"] >= pd.Timestamp(d_start)) & (df["Date"] <= pd.Timestamp(d_end))]

    # Candlestick + Volume
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.04,
    )
    fig.add_trace(go.Candlestick(
        x=view["Date"], open=view["Open"], high=view["High"],
        low=view["Low"], close=view["Close"],
        increasing_line_color="#00c88c", decreasing_line_color="#ff4d6d",
        name="OHLC",
    ), row=1, col=1)

    # MA overlays
    for ma_col, color, width in [("sma_20","#fbbf24",1.5), ("ema_26","#818cf8",1.5)]:
        if ma_col in view.columns:
            fig.add_trace(go.Scatter(
                x=view["Date"], y=view[ma_col], mode="lines",
                line=dict(color=color, width=width), name=ma_col.upper(),
            ), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in view.columns:
        fig.add_trace(go.Scatter(
            x=view["Date"], y=view["bb_upper"],
            mode="lines", line=dict(color="rgba(130,80,255,0.4)", width=1),
            name="BB Upper", showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=view["Date"], y=view["bb_lower"],
            mode="lines", line=dict(color="rgba(130,80,255,0.4)", width=1),
            fill="tonexty", fillcolor="rgba(130,80,255,0.05)",
            name="BB Lower", showlegend=True,
        ), row=1, col=1)

    # Volume bars
    colors = ["#00c88c" if r >= 0 else "#ff4d6d" for r in view["daily_return"].fillna(0)]
    fig.add_trace(go.Bar(
        x=view["Date"], y=view["Volume"], marker_color=colors,
        name="Volume", opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)
    _plotly_dark_layout(fig, f"{selected_ticker} — Price & Volume")
    st.plotly_chart(fig, width='stretch')

    # RSI chart
    st.markdown("<div class='section-title'>RSI (14-day)</div>", unsafe_allow_html=True)
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(
        x=view["Date"], y=view["rsi"], mode="lines",
        line=dict(color="#4f8ef7", width=2), name="RSI",
    ))
    rsi_fig.add_hline(y=70, line=dict(color="#ff4d6d", dash="dash"), annotation_text="Overbought (70)")
    rsi_fig.add_hline(y=30, line=dict(color="#00c88c", dash="dash"), annotation_text="Oversold (30)")
    _plotly_dark_layout(rsi_fig, "Relative Strength Index")
    st.plotly_chart(rsi_fig, width='stretch')

    # MACD
    st.markdown("<div class='section-title'>MACD</div>", unsafe_allow_html=True)
    macd_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.04)
    macd_fig.add_trace(go.Scatter(x=view["Date"], y=view["macd"], mode="lines",
        line=dict(color="#4f8ef7",width=2), name="MACD"), row=1, col=1)
    macd_fig.add_trace(go.Scatter(x=view["Date"], y=view["macd_signal"], mode="lines",
        line=dict(color="#f59e0b",width=1.5), name="Signal"), row=1, col=1)
    hist_colors = ["#00c88c" if v >= 0 else "#ff4d6d" for v in view["macd_histogram"].fillna(0)]
    macd_fig.add_trace(go.Bar(x=view["Date"], y=view["macd_histogram"],
        marker_color=hist_colors, name="Histogram"), row=2, col=1)
    _plotly_dark_layout(macd_fig, "MACD Indicator")
    st.plotly_chart(macd_fig, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Sentiment Trends
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Sentiment Trends":
    st.markdown(f"<div class='section-title'>💬 Sentiment Trends — {selected_ticker}</div>", unsafe_allow_html=True)

    sent_cols = ["avg_sentiment", "pos_count", "neg_count", "neu_count", "sentiment_volatility"]
    has_sent  = all(c in df.columns for c in sent_cols)

    if not has_sent:
        st.warning("Sentiment columns not found in the feature dataset. "
                   "Ensure the sentiment pipeline was run and merged before feature engineering.")
        st.stop()

    view = df.copy()

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    avg_s = view["avg_sentiment"].mean()
    pos_t = int(view["pos_count"].sum())
    neg_t = int(view["neg_count"].sum())
    sv    = view["sentiment_volatility"].mean()

    for col, val, label in zip(
        [c1, c2, c3, c4],
        [f"{avg_s:+.3f}", pos_t, neg_t, f"{sv:.3f}"],
        ["Avg Daily Sentiment", "Total Positive News", "Total Negative News", "Avg Sentiment Volatility"],
    ):
        with col:
            st.markdown(f"""<div class="metric-card"><h2>{val}</h2><p>{label}</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Sentiment time series
    fig_sent = go.Figure()
    fig_sent.add_trace(go.Scatter(
        x=view["Date"], y=view["avg_sentiment"].rolling(7).mean(),
        mode="lines", line=dict(color="#4f8ef7", width=2.5),
        name="7-day MA Sentiment",
    ))
    fig_sent.add_trace(go.Scatter(
        x=view["Date"], y=view["avg_sentiment"],
        mode="lines", line=dict(color="rgba(79,142,247,0.3)", width=1),
        name="Daily Sentiment",
    ))
    fig_sent.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", dash="dash"))
    _plotly_dark_layout(fig_sent, "Daily Average Sentiment Score (7-day Rolling)")
    st.plotly_chart(fig_sent, width='stretch')

    # Stacked bar — pos / neu / neg counts
    st.markdown("<div class='section-title'>News Count by Sentiment</div>", unsafe_allow_html=True)
    # Resample to weekly for readability
    weekly = view.set_index("Date").resample("W")[["pos_count","neg_count","neu_count"]].sum().reset_index()
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=weekly["Date"], y=weekly["pos_count"],
        name="Positive", marker_color="#00c88c"))
    bar_fig.add_trace(go.Bar(x=weekly["Date"], y=weekly["neg_count"],
        name="Negative", marker_color="#ff4d6d"))
    bar_fig.add_trace(go.Bar(x=weekly["Date"], y=weekly["neu_count"],
        name="Neutral",  marker_color="#94a3b8"))
    bar_fig.update_layout(barmode="stack")
    _plotly_dark_layout(bar_fig, "Weekly News Count by Sentiment Category")
    st.plotly_chart(bar_fig, width='stretch')

    # Sentiment vs Price scatter
    st.markdown("<div class='section-title'>Sentiment vs Next-Day Price Direction</div>", unsafe_allow_html=True)
    scatter_fig = px.scatter(
        view, x="avg_sentiment", y="daily_return",
        color="target", color_discrete_map={1: "#00c88c", 0: "#ff4d6d"},
        labels={"avg_sentiment": "Average Sentiment", "daily_return": "Daily Return",
                "target": "Next Day"},
        hover_data=["Date"],
    )
    scatter_fig.update_traces(marker=dict(size=5, opacity=0.7))
    _plotly_dark_layout(scatter_fig, "Avg Sentiment vs Daily Return")
    st.plotly_chart(scatter_fig, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Prediction Engine
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction Engine":
    st.markdown(f"<div class='section-title'>🔮 Prediction Engine — {selected_ticker} / {selected_model}</div>", unsafe_allow_html=True)

    if model is None:
        st.error(f"Model **{selected_model}** not found for **{selected_ticker}**. "
                 "Please train the models using `notebooks/04_model_training.ipynb`.")
        st.stop()

    # Date selector
    available_dates = df["Date"].dt.date.tolist()
    selected_date   = st.date_input(
        "Select a trading day to predict",
        value=available_dates[-1],
        min_value=available_dates[0],
        max_value=available_dates[-1],
    )

    row = df[df["Date"].dt.date == selected_date]
    if row.empty:
        st.warning("No data found for the selected date. Try an adjacent trading day.")
        st.stop()

    available_features = [c for c in ALL_FEATURES if c in df.columns]
    X_row = row[available_features]

    prob  = model.predict_proba(X_row)[0]
    pred  = model.predict(X_row)[0]
    conf  = max(prob)
    actual = int(row["target"].values[0]) if "target" in row.columns else None

    # Prediction banner
    st.markdown("")
    if pred == 1:
        st.markdown(f"<div class='pred-up'>📈 PRICE UP ↑<br><small style='font-size:1rem;'>Confidence: {conf*100:.1f}%</small></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='pred-down'>📉 PRICE DOWN ↓<br><small style='font-size:1rem;'>Confidence: {conf*100:.1f}%</small></div>", unsafe_allow_html=True)

    st.markdown("")

    if actual is not None:
        correct = (pred == actual)
        st.info(f"**Actual outcome:** {'↑ UP' if actual == 1 else '↓ DOWN'}  |  "
                f"**Prediction:** {'✅ Correct' if correct else '❌ Incorrect'}")

    # Probability gauge
    col_g, col_f = st.columns([1, 1])
    with col_g:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob[1] * 100,
            number={"suffix": "%", "font": {"color": "white", "size": 40}},
            gauge={
                "axis":      {"range": [0, 100], "tickcolor": "#6b7280"},
                "bar":       {"color": "#4f8ef7"},
                "bgcolor":   "rgba(0,0,0,0)",
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0,  40], "color": "rgba(255,77,109,0.3)"},
                    {"range": [40, 60], "color": "rgba(148,163,184,0.2)"},
                    {"range": [60,100], "color": "rgba(0,200,140,0.3)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": 50},
            },
            title={"text": "P(Price Up)", "font": {"color": "#c0c0d0"}},
        ))
        gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=250,
                            margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(gauge, width='stretch')

    with col_f:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Key features on this date:**")
        feat_display = {
            "Close":            f"${row['Close'].values[0]:.2f}",
            "RSI":              f"{row['rsi'].values[0]:.1f}" if "rsi" in row.columns else "—",
            "MACD":             f"{row['macd'].values[0]:.4f}" if "macd" in row.columns else "—",
            "SMA 20":           f"${row['sma_20'].values[0]:.2f}" if "sma_20" in row.columns else "—",
            "Avg Sentiment":    f"{row['avg_sentiment'].values[0]:+.3f}" if "avg_sentiment" in row.columns else "—",
            "Sentiment Vol.":   f"{row['sentiment_volatility'].values[0]:.3f}" if "sentiment_volatility" in row.columns else "—",
        }
        feat_df = pd.DataFrame.from_dict(feat_display, orient="index", columns=["Value"])
        st.dataframe(feat_df, width='stretch')

    # Probability bar
    st.markdown("---")
    prob_fig = go.Figure(go.Bar(
        x=["↓ Down", "↑ Up"], y=[prob[0]*100, prob[1]*100],
        marker_color=["#ff4d6d","#00c88c"],
        text=[f"{prob[0]*100:.1f}%", f"{prob[1]*100:.1f}%"],
        textposition="outside", textfont=dict(color="white", size=16),
    ))
    prob_fig.update_layout(yaxis_range=[0, 115])
    _plotly_dark_layout(prob_fig, "Prediction Probability Distribution")
    st.plotly_chart(prob_fig, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Explainability
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Explainability":
    st.markdown(f"<div class='section-title'>🔍 Model Explainability — {selected_model}</div>", unsafe_allow_html=True)

    if model is None:
        st.error("Model not found. Train models first.")
        st.stop()

    import shap
    import matplotlib.pyplot as plt

    available_features = [c for c in ALL_FEATURES if c in df.columns]

    # Split test set for SHAP
    test_df = df[df["Date"].dt.year >= 2023]
    if len(test_df) < 10:
        test_df = df.tail(200)

    train_df  = df[df["Date"] < test_df["Date"].min()]
    X_test_s  = test_df[available_features]
    X_train_s = train_df[available_features]

    tab1, tab2, tab3 = st.tabs(["🌐 Global Feature Importance", "🔬 Local Explanation", "📊 Feature Comparison"])

    with tab1:
        st.markdown("**Which features matter most across all predictions?**")
        with st.spinner("Computing SHAP values …"):
            try:
                from src.explainer import compute_shap_values, plot_global_importance
                sv     = compute_shap_values(model, X_test_s, X_background=X_train_s)
                g_fig  = plot_global_importance(sv, available_features, model_name=selected_model, save=False)
                st.pyplot(g_fig, width='stretch')
            except Exception as e:
                st.error(f"SHAP computation failed: {e}")
                st.info("This can happen if the model is not compatible. Try another model.")

    with tab2:
        st.markdown("**How did each feature contribute to a specific prediction?**")
        sample_idx = st.slider("Sample index (test set)", 0, min(len(X_test_s)-1, 100), 0)

        with st.spinner("Computing local SHAP values …"):
            try:
                from src.explainer import compute_shap_values, plot_waterfall
                sv    = compute_shap_values(model, X_test_s.iloc[:sample_idx+5], X_background=X_train_s)
                w_fig = plot_waterfall(sv, sample_idx, available_features,
                                       model_name=selected_model, save=False)
                st.pyplot(w_fig, width='stretch')
            except Exception as e:
                st.error(f"Waterfall plot failed: {e}")

    with tab3:
        st.markdown("**Feature importance across all four models:**")

        # Load models and compare importances
        importance_rows = []
        for mn in MODEL_NAMES:
            m = load_model(selected_ticker, mn)
            if m is None:
                continue
            inner = m.named_steps["clf"] if hasattr(m, "named_steps") else m
            if hasattr(inner, "feature_importances_"):
                fi = inner.feature_importances_
                for feat, imp in zip(available_features, fi[:len(available_features)]):
                    importance_rows.append({"Model": mn, "Feature": feat, "Importance": imp})
            elif hasattr(inner, "coef_"):
                fi = np.abs(inner.coef_[0])
                for feat, imp in zip(available_features, fi[:len(available_features)]):
                    importance_rows.append({"Model": mn, "Feature": feat, "Importance": imp})

        if importance_rows:
            imp_df  = pd.DataFrame(importance_rows)
            top_10  = imp_df.groupby("Feature")["Importance"].mean().nlargest(10).index.tolist()
            imp_sub = imp_df[imp_df["Feature"].isin(top_10)]

            comp_fig = px.bar(
                imp_sub, x="Importance", y="Feature", color="Model",
                orientation="h", barmode="group",
                color_discrete_sequence=["#4f8ef7","#00c88c","#f59e0b","#a78bfa"],
            )
            _plotly_dark_layout(comp_fig, "Top-10 Feature Importance Comparison Across Models")
            st.plotly_chart(comp_fig, width='stretch')
        else:
            st.info("Train all four models to see the comparison.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Model Performance":
    st.markdown(f"<div class='section-title'>📉 Model Performance — {selected_ticker}</div>", unsafe_allow_html=True)

    from sklearn.metrics import roc_curve, auc, confusion_matrix
    from src.utils import VAL_END
    from src.feature_engineer import ALL_FEATURES

    available_features = [c for c in ALL_FEATURES if c in df.columns]
    test_df = df[df["Date"] > VAL_END].copy()

    if test_df.empty:
        st.warning("No test data found. Ensure the date range covers 2023-2024.")
        st.stop()

    X_test = test_df[available_features]
    y_test = test_df["target"]

    metrics_rows = []
    roc_traces   = []
    cm_data      = {}

    for mn in MODEL_NAMES:
        m = load_model(selected_ticker, mn)
        if m is None:
            continue
        y_pred  = m.predict(X_test)
        y_proba = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else None

        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score, roc_auc_score)
        metrics_rows.append({
            "Model":     mn,
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan"), 4),
        })
        cm_data[mn] = confusion_matrix(y_test, y_pred)

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_traces.append((mn, fpr, tpr, auc(fpr, tpr)))

    if not metrics_rows:
        st.warning("No trained models found for this ticker. Run the model training notebook first.")
        st.stop()

    metrics_df = pd.DataFrame(metrics_rows).set_index("Model")

    # Metrics table
    st.markdown("<div class='section-title'>Evaluation Metrics (Test Set)</div>", unsafe_allow_html=True)
    st.dataframe(
        metrics_df.style.background_gradient(cmap="Blues", axis=0),
        width='stretch',
    )

    # Highlight best
    best_model_name = metrics_df["F1 Score"].idxmax()
    st.success(f"🏆 **Best Model by F1 Score:** {best_model_name} "
               f"({metrics_df.loc[best_model_name,'F1 Score']:.4f})")

    st.markdown("---")

    # ROC Curves
    if roc_traces:
        st.markdown("<div class='section-title'>ROC Curves</div>", unsafe_allow_html=True)
        roc_colors = ["#4f8ef7", "#00c88c", "#f59e0b", "#a78bfa"]
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
            name="Random", showlegend=True,
        ))
        for (mn, fpr, tpr, roc_auc), color in zip(roc_traces, roc_colors):
            roc_fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color=color, width=2.5),
                name=f"{mn} (AUC={roc_auc:.3f})",
            ))
        _plotly_dark_layout(roc_fig, "ROC Curves — All Models")
        st.plotly_chart(roc_fig, width='stretch')

    # Confusion matrices
    if cm_data:
        st.markdown("<div class='section-title'>Confusion Matrices</div>", unsafe_allow_html=True)
        cols = st.columns(min(len(cm_data), 4))
        for idx, (mn, cm) in enumerate(cm_data.items()):
            with cols[idx % 4]:
                cm_fig = px.imshow(
                    cm, text_auto=True, color_continuous_scale="Blues",
                    labels={"x":"Predicted","y":"Actual"},
                    x=["Down","Up"], y=["Down","Up"],
                )
                cm_fig.update_layout(
                    title=dict(text=mn, font=dict(color="white")),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    height=280,
                    margin=dict(l=10,r=10,t=40,b=10),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(cm_fig, width='stretch')

    # Bar chart comparison
    st.markdown("<div class='section-title'>Metrics Bar Comparison</div>", unsafe_allow_html=True)
    bar_metrics = metrics_df[["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]].reset_index()
    bar_melt = bar_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
    bar_cmp = px.bar(
        bar_melt, x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_sequence=["#4f8ef7","#00c88c","#f59e0b","#a78bfa"],
        range_y=[0, 1.05],
    )
    _plotly_dark_layout(bar_cmp, "Model Comparison by Metric")
    st.plotly_chart(bar_cmp, width='stretch')

