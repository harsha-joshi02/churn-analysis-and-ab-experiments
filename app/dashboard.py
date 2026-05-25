"""
Churn Intelligence Platform — Streamlit Dashboard
──────────────────────────────────────────────────
Pages:
  1. Churn Overview    — KPIs and segment breakdown
  2. Model Performance — ROC, PR, SHAP, feature importance
  3. Customer Explorer — per-customer risk + SHAP drivers
  4. Experiments       — Bayesian A/B test runner & history
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

# ── Path setup so imports resolve from project root ───────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.utils import MODEL_PATH, PLOTS_DIR, TEST_PREDS_PATH
from src.predict import load_model, load_predictions, segment_summary, get_top_churn_drivers, RISK_BINS, RISK_LABELS
from src.experiments import (
    ExperimentData,
    BayesianResult,
    run_bayesian_ab_test,
    simulate_discount_experiment,
    save_experiment_to_db,
    get_all_experiments,
)
from src.db import is_db_available, init_db


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar brand */
    [data-testid="stSidebar"] { background: #0d1b2a; }
    [data-testid="stSidebar"] * { color: #e0e6ed !important; }

    /* KPI metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1b2a3b 0%, #0d1b2a 100%);
        border: 1px solid #2e4057;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    div[data-testid="metric-container"] label {
        font-size: 0.78rem !important;
        color: #8da0b5 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #4ec9b0 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.82rem !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4ec9b0;
        border-left: 4px solid #4ec9b0;
        padding-left: 10px;
        margin: 18px 0 10px 0;
    }

    /* Risk badge colours */
    .badge-high   { color: #ff6b6b; font-weight: 700; }
    .badge-medium { color: #ffd166; font-weight: 700; }
    .badge-low    { color: #06d6a0; font-weight: 700; }

    /* Horizontal rule */
    hr { border-color: #2e4057; }
    </style>
    """,
    unsafe_allow_html=True,
)

SEGMENT_COLORS = {"High": "#ff6b6b", "Medium": "#ffd166", "Low": "#06d6a0"}


# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def _load_model():
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner="Loading predictions …", ttl=300)
def _load_preds() -> pd.DataFrame:
    df = load_predictions(TEST_PREDS_PATH)
    if "risk_segment" not in df.columns:
        df["risk_segment"] = pd.cut(
            df["churn_probability"], bins=RISK_BINS, labels=RISK_LABELS, right=False
        )
    return df


def _model_ready() -> bool:
    return MODEL_PATH.exists() and TEST_PREDS_PATH.exists()


def _setup_warning():
    st.warning(
        "**Model not trained yet.**  Run the setup pipeline first:\n\n"
        "```bash\n"
        "cd churn_platform\n"
        "python scripts/generate_data.py\n"
        "python -m src.train\n"
        "```",
        icon="⚠️",
    )


def _render_experiment_result(exp_data: ExperimentData, result: BayesianResult) -> None:
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(Treatment > Control)", f"{result.prob_treatment_beats_control:.1%}")
    c2.metric("Expected Lift", f"{result.expected_lift:.1%}")
    c3.metric("Control Retention Rate", f"{result.control_rate:.1%}")
    c4.metric("Treatment Retention Rate", f"{result.treatment_rate:.1%}")

    st.markdown(
        f"**Recommended sample size per arm:** {result.recommended_sample_size:,}  \n"
        f"Control: {exp_data.control_conversions}/{exp_data.control_trials} retained  |  "
        f"Treatment: {exp_data.treatment_conversions}/{exp_data.treatment_trials} retained"
    )

    ctrl_samples = np.array(result.control_posterior["samples"])
    trt_samples = np.array(result.treatment_posterior["samples"])

    fig_post = go.Figure()
    fig_post.add_trace(
        go.Histogram(
            x=ctrl_samples,
            nbinsx=60,
            name="Control (no intervention)",
            opacity=0.65,
            marker_color="#4C72B0",
            histnorm="probability density",
        )
    )
    fig_post.add_trace(
        go.Histogram(
            x=trt_samples,
            nbinsx=60,
            name="Treatment (discount)",
            opacity=0.65,
            marker_color="#DD8452",
            histnorm="probability density",
        )
    )
    fig_post.update_layout(
        barmode="overlay",
        xaxis_title="Retention Rate θ",
        yaxis_title="Posterior Density",
        title=f"Posterior Beta distributions — P(Treatment > Control) = {result.prob_treatment_beats_control:.1%}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d6e3",
        legend=dict(x=0.01, y=0.95),
        height=380,
    )
    st.plotly_chart(fig_post, use_container_width=True)

    p = result.prob_treatment_beats_control
    if p >= 0.95:
        st.success(f"**Strong evidence** that the intervention works (P = {p:.1%}). Consider rolling out to all High-risk customers.")
    elif p >= 0.80:
        st.info(f"**Promising signal** (P = {p:.1%}). Collect more data before a full rollout.")
    else:
        st.warning(f"**Inconclusive** (P = {p:.1%}). The intervention may not justify the discount cost.")


def _render_history_row(exp: dict, row_key: str) -> None:
    st.markdown(
        f"**Description:** {exp.get('description', '—')}  \n"
        f"**Segment:** {exp.get('segment', '—')}  \n"
        f"**Control:** {exp.get('control_description', '—')}  \n"
        f"**Treatment:** {exp.get('treatment_description', '—')}"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(trt > ctrl)", f"{exp['prob_treatment_beats_control']:.1%}")
    c2.metric("Expected Lift", f"{exp['expected_lift']:.1%}")
    c3.metric("Recommended N/arm", f"{exp['recommended_sample_size']:,}")
    c4.metric("Sample", f"{exp['control_trials']} ctrl / {exp['treatment_trials']} trt")

    posterior = exp.get("posterior_data", {})
    if posterior:
        ctrl_s = np.array(posterior.get("control_samples", []))
        trt_s = np.array(posterior.get("treatment_samples", []))
        if len(ctrl_s) and len(trt_s):
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=ctrl_s, nbinsx=50, name="Control",
                                       opacity=0.65, marker_color="#4C72B0", histnorm="probability density"))
            fig.add_trace(go.Histogram(x=trt_s, nbinsx=50, name="Treatment",
                                       opacity=0.65, marker_color="#DD8452", histnorm="probability density"))
            fig.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d6e3",
                              height=280, margin=dict(t=20, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"history_posterior_{row_key}")


# ── Sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "### 📊 Churn Intelligence\n"
        "<small style='color:#8da0b5'>SaaS Retention Platform</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠  Churn Overview",
            "📈  Model Performance",
            "🔍  Customer Explorer",
            "🧪  Experiments",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#556880'>"
        "XGBoost · Optuna · MLflow<br>"
        "Bayesian A/B · SHAP<br>"
        "PostgreSQL · Streamlit"
        "</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CHURN OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠  Churn Overview":
    st.title("Churn Overview")
    st.markdown("*High-level KPIs and segment breakdown across the full customer base.*")

    if not _model_ready():
        _setup_warning()
        st.stop()

    preds = _load_preds()

    total = len(preds)
    churn_rate = preds["churn_probability"].mean()
    high_risk_n = int((preds["risk_segment"] == "High").sum())
    med_risk_n = int((preds["risk_segment"] == "Medium").sum())
    avg_tenure = preds["tenure"].mean()
    avg_charge = preds["monthly_charges"].mean()
    monthly_at_risk = preds.loc[preds["risk_segment"] == "High", "monthly_charges"].sum()

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Avg Churn Prob", f"{churn_rate:.1%}")
    c3.metric("High Risk", f"{high_risk_n:,}", f"{high_risk_n/total:.1%} of base")
    c4.metric("Medium Risk", f"{med_risk_n:,}", f"{med_risk_n/total:.1%} of base")
    c5.metric("Avg Tenure (mo)", f"{avg_tenure:.1f}")
    c6.metric("Monthly Revenue at Risk", f"${monthly_at_risk:,.0f}")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    # ── Segment doughnut ─────────────────────────────────────────────────────
    with col_left:
        st.markdown('<p class="section-header">Customer Risk Distribution</p>', unsafe_allow_html=True)
        seg_counts = preds["risk_segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_pie = px.pie(
            seg_counts,
            names="Segment",
            values="Count",
            hole=0.55,
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d6e3",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Avg KPIs by segment ───────────────────────────────────────────────────
    with col_right:
        st.markdown('<p class="section-header">Segment-Level KPIs</p>', unsafe_allow_html=True)
        summary = segment_summary(preds)
        for _, row in summary.iterrows():
            seg = row["risk_segment"]
            badge_cls = f"badge-{seg.lower()}"
            st.markdown(
                f"<b><span class='{badge_cls}'>{seg} Risk</span></b> &nbsp;"
                f"| Customers: **{int(row['count']):,}** "
                f"| Avg churn prob: **{row['avg_churn_prob']:.1%}** "
                f"| Avg tenure: **{row['avg_tenure']:.0f} mo** "
                f"| Avg charge: **${row['avg_monthly_charges']:.0f}/mo**",
                unsafe_allow_html=True,
            )
            st.markdown("")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    # ── Churn probability histogram ───────────────────────────────────────────
    with col_a:
        st.markdown('<p class="section-header">Churn Probability Distribution</p>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            preds,
            x="churn_probability",
            nbins=40,
            color="risk_segment",
            color_discrete_map=SEGMENT_COLORS,
            labels={"churn_probability": "Churn Probability", "risk_segment": "Segment"},
            opacity=0.85,
        )
        fig_hist.add_vline(x=0.30, line_dash="dash", line_color="#ffd166", annotation_text="Medium threshold")
        fig_hist.add_vline(x=0.60, line_dash="dash", line_color="#ff6b6b", annotation_text="High threshold")
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d6e3",
            barmode="overlay",
            legend_title="Segment",
            height=320,
            margin=dict(t=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Avg tenure by segment bar chart ──────────────────────────────────────
    with col_b:
        st.markdown('<p class="section-header">Average Tenure by Segment</p>', unsafe_allow_html=True)
        fig_bar = px.bar(
            summary.sort_values("avg_tenure"),
            x="risk_segment",
            y="avg_tenure",
            color="risk_segment",
            color_discrete_map=SEGMENT_COLORS,
            text_auto=".1f",
            labels={"risk_segment": "Risk Segment", "avg_tenure": "Avg Tenure (months)"},
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d6e3",
            showlegend=False,
            height=320,
            margin=dict(t=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Contract type × churn ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">Churn Rate by Contract Type</p>', unsafe_allow_html=True)
    if "contract_type" in preds.columns:
        ct_grp = (
            preds.groupby("contract_type")["churn_probability"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_churn_prob", "count": "n_customers"})
            .sort_values("avg_churn_prob", ascending=False)
        )
        fig_ct = px.bar(
            ct_grp,
            x="contract_type",
            y="avg_churn_prob",
            text_auto=".1%",
            color="avg_churn_prob",
            color_continuous_scale="RdYlGn_r",
            labels={"contract_type": "Contract Type", "avg_churn_prob": "Avg Churn Prob"},
        )
        fig_ct.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d6e3",
            coloraxis_showscale=False,
            height=280,
            margin=dict(t=10),
        )
        st.plotly_chart(fig_ct, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈  Model Performance":
    st.title("Model Performance")
    st.markdown("*XGBoost trained with Optuna HPO · Evaluated on held-out test set*")

    if not _model_ready():
        _setup_warning()
        st.stop()

    preds = _load_preds()
    model, feature_names = _load_model()

    # Use churn_probability vs y_true for metric curves
    y_true = preds["y_true"].values
    y_score = preds["churn_probability"].values

    auc = roc_auc_score(y_true, y_score)

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC", f"{auc:.4f}")

    from sklearn.metrics import f1_score, average_precision_score
    y_pred = (y_score >= 0.5).astype(int)
    col2.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
    col3.metric("Avg Precision", f"{average_precision_score(y_true, y_score):.4f}")

    st.markdown("---")

    tab_roc, tab_pr, tab_fi, tab_shap = st.tabs(
        ["ROC Curve", "PR Curve", "Feature Importance", "SHAP Summary"]
    )

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    with tab_roc:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"XGBoost (AUC = {auc:.3f})",
                line=dict(color="#4ec9b0", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(78,201,176,0.08)",
            )
        )
        fig_roc.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(dash="dash", color="#556880"),
        )
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d6e3",
            legend=dict(x=0.6, y=0.1),
            height=420,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── PR Curve ──────────────────────────────────────────────────────────────
    with tab_pr:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, y_score)
        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=recall, y=precision,
                mode="lines",
                name=f"XGBoost (AP = {ap:.3f})",
                line=dict(color="#f4a261", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(244,162,97,0.08)",
            )
        )
        baseline = y_true.mean()
        fig_pr.add_shape(
            type="line", x0=0, y0=baseline, x1=1, y1=baseline,
            line=dict(dash="dash", color="#556880"),
            name="Baseline",
        )
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d6e3",
            legend=dict(x=0.6, y=0.9),
            height=420,
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    with tab_fi:
        importances = model.feature_importances_
        fi_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=True)
            .tail(15)
        )
        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Teal",
            labels={"Importance": "XGBoost Gain"},
        )
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d6e3",
            coloraxis_showscale=False,
            height=460,
            margin=dict(l=130),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── SHAP Summary ──────────────────────────────────────────────────────────
    with tab_shap:
        shap_img = PLOTS_DIR / "shap_summary.png"
        if shap_img.exists():
            st.image(str(shap_img), caption="SHAP Summary Plot (300-sample subset)", use_column_width=True)
        else:
            st.info("Run training to generate the SHAP summary plot.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍  Customer Explorer":
    st.title("Customer Explorer")
    st.markdown("*Filter, browse, and drill into individual churn risk profiles.*")

    if not _model_ready():
        _setup_warning()
        st.stop()

    preds = _load_preds()
    model, feature_names = _load_model()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filters")
        seg_filter = st.multiselect(
            "Risk Segment",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
        tenure_range = st.slider("Tenure (months)", 0, int(preds["tenure"].max()), (0, int(preds["tenure"].max())))
        charge_range = st.slider(
            "Monthly Charges ($)",
            float(preds["monthly_charges"].min()),
            float(preds["monthly_charges"].max()),
            (float(preds["monthly_charges"].min()), float(preds["monthly_charges"].max())),
        )

    filtered = preds[
        preds["risk_segment"].isin(seg_filter)
        & preds["tenure"].between(*tenure_range)
        & preds["monthly_charges"].between(*charge_range)
    ].copy()

    st.markdown(f"**{len(filtered):,}** customers match the current filters.")

    # ── Customer table ────────────────────────────────────────────────────────
    display_cols = [
        "customer_id", "risk_segment", "churn_probability",
        "tenure", "monthly_charges", "total_charges",
        "contract_type", "internet_service", "senior_citizen",
    ]
    show_cols = [c for c in display_cols if c in filtered.columns]

    table_df = filtered[show_cols].sort_values("churn_probability", ascending=False).reset_index(drop=True)
    table_df["churn_probability"] = table_df["churn_probability"].map("{:.1%}".format)

    st.dataframe(
        table_df.head(200),
        use_container_width=True,
        height=320,
        column_config={
            "customer_id": "Customer ID",
            "risk_segment": st.column_config.TextColumn("Risk", width="small"),
            "churn_probability": "Churn Prob",
            "tenure": st.column_config.NumberColumn("Tenure (mo)", format="%d"),
            "monthly_charges": st.column_config.NumberColumn("Monthly $", format="$%.2f"),
            "total_charges": st.column_config.NumberColumn("Total Charges $", format="$%.0f"),
            "senior_citizen": st.column_config.NumberColumn("Senior", format="%d"),
        },
    )

    st.markdown("---")
    st.markdown('<p class="section-header">Individual Customer Risk Profile</p>', unsafe_allow_html=True)

    # ── Single customer drilldown ─────────────────────────────────────────────
    from src.features import load_raw_data
    raw_df = load_raw_data()

    customer_options = filtered["customer_id"].tolist()
    if not customer_options:
        st.info("No customers match the current filters.")
        st.stop()

    selected_id = st.selectbox("Select Customer", customer_options[:500])

    if selected_id:
        cust_row = preds[preds["customer_id"] == selected_id].iloc[0]
        churn_prob = float(cust_row["churn_probability"])
        segment = str(cust_row["risk_segment"])

        col_gauge, col_info = st.columns([1, 2])

        with col_gauge:
            # Gauge chart for churn probability
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=churn_prob * 100,
                    number={"suffix": "%", "font": {"size": 36, "color": SEGMENT_COLORS.get(segment, "#fff")}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8da0b5"},
                        "bar": {"color": SEGMENT_COLORS.get(segment, "#4ec9b0")},
                        "bgcolor": "#1b2a3b",
                        "bordercolor": "#2e4057",
                        "steps": [
                            {"range": [0, 30], "color": "rgba(6,214,160,0.15)"},
                            {"range": [30, 60], "color": "rgba(255,209,102,0.15)"},
                            {"range": [60, 100], "color": "rgba(255,107,107,0.15)"},
                        ],
                        "threshold": {
                            "line": {"color": "#fff", "width": 3},
                            "thickness": 0.8,
                            "value": churn_prob * 100,
                        },
                    },
                    title={"text": f"Churn Risk · {segment}", "font": {"size": 14, "color": "#8da0b5"}},
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d6e3",
                height=260,
                margin=dict(t=30, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_info:
            st.markdown(f"**Customer ID:** `{selected_id}`")
            for col in ["tenure", "monthly_charges", "contract_type",
                        "num_products", "total_charges", "senior_citizen",
                        "partner", "dependents", "paperless_billing",
                        "payment_method", "internet_service"]:
                if col in cust_row.index:
                    val = cust_row[col]
                    label = col.replace("_", " ").title()
                    st.markdown(f"- **{label}:** {val}")

        # ── SHAP drivers ──────────────────────────────────────────────────────
        st.markdown("**Top Churn Drivers (SHAP)**")
        try:
            drivers = get_top_churn_drivers(
                raw_df, selected_id, model=model, feature_names=feature_names, top_n=8
            )
            drivers["direction"] = drivers["shap_value"].apply(
                lambda v: "↑ Increases risk" if v > 0 else "↓ Decreases risk"
            )
            fig_drivers = px.bar(
                drivers.sort_values("shap_value"),
                x="shap_value",
                y="feature",
                orientation="h",
                color="shap_value",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                text="direction",
                labels={"shap_value": "SHAP value (impact on log-odds)", "feature": "Feature"},
            )
            fig_drivers.update_traces(textposition="outside")
            fig_drivers.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d6e3",
                coloraxis_showscale=False,
                height=340,
                margin=dict(l=140),
            )
            st.plotly_chart(fig_drivers, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not compute SHAP drivers: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧪  Experiments":
    st.title("Bayesian A/B Experiments")
    st.markdown(
        "*Model customer retention interventions using Bayesian updating "
        "on Beta-Binomial conjugate priors.*"
    )

    db_ok = is_db_available()
    if not db_ok:
        st.warning(
            "PostgreSQL is not reachable — experiments will run in-memory only "
            "and won't be persisted. Start the database with `docker compose up postgres`.",
            icon="⚠️",
        )

    if not _model_ready():
        _setup_warning()
        st.stop()

    preds = _load_preds()

    tab_run, tab_history = st.tabs(["▶ Run Experiment", "📋 Experiment History"])

    # ── Run experiment tab ────────────────────────────────────────────────────
    with tab_run:
        st.markdown('<p class="section-header">Configure Experiment</p>', unsafe_allow_html=True)

        with st.form("experiment_form"):
            exp_name = st.text_input("Experiment Name", value="discount_intervention_high_risk")
            exp_desc = st.text_area(
                "Description",
                value="20% monthly discount offered to High-risk customers for 3 months vs no intervention.",
                height=80,
            )
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                target_segment = st.selectbox("Target Segment", ["High", "Medium", "Low"], index=0)
                ctrl_desc = st.text_input("Control Description", "No intervention")
                trt_desc = st.text_input("Treatment Description", "20% discount for 3 months")

            with col_f2:
                discount_lift = st.number_input(
                    "Assumed Retention Lift from Intervention (%)",
                    min_value=0.01,
                    max_value=99.0,
                    value=15.0,
                    step=0.01,
                    format="%.2f",
                    help="Observed or assumed % of additional customers retained due to the intervention.",
                )
                prior_alpha = st.number_input("Prior α (Beta prior)", value=1.0, min_value=0.1, step=0.5)
                prior_beta = st.number_input("Prior β (Beta prior)", value=1.0, min_value=0.1, step=0.5)

            submitted = st.form_submit_button("🚀 Run Experiment", type="primary")

        if submitted:
            cohort = preds[preds["risk_segment"] == target_segment]
            if len(cohort) < 10:
                st.error(f"Not enough customers in the '{target_segment}' segment ({len(cohort)}).")
            else:
                with st.spinner("Running Bayesian simulation …"):
                    exp_data, result = simulate_discount_experiment(
                        cohort, discount_lift_pct=discount_lift / 100.0
                    )
                    # Override with form values
                    exp_data.name = exp_name or exp_data.name
                    exp_data.description = exp_desc or exp_data.description
                    exp_data.control_description = ctrl_desc
                    exp_data.treatment_description = trt_desc
                    exp_data.prior_alpha = prior_alpha
                    exp_data.prior_beta = prior_beta

                    if db_ok:
                        try:
                            init_db()
                            save_experiment_to_db(exp_data, result)
                        except Exception as exc:
                            st.warning(f"Could not persist to DB: {exc}")

                st.success("Experiment complete!")
                _render_experiment_result(exp_data, result)  # defined below

    # ── History tab ───────────────────────────────────────────────────────────
    with tab_history:
        if not db_ok:
            st.info("Database not available — no persistent history to display.")
        else:
            try:
                init_db()
                history = get_all_experiments()
            except Exception:
                history = []

            if not history:
                st.info("No experiments recorded yet. Run one from the **▶ Run Experiment** tab.")
            else:
                for i, exp in enumerate(history):
                    with st.expander(
                        f"🧪 {exp['name']}  |  P(trt > ctrl) = {exp['prob_treatment_beats_control']:.1%}  "
                        f"|  {exp['created_at'].strftime('%Y-%m-%d %H:%M') if exp.get('created_at') else ''}",
                        expanded=False,
                    ):
                        _render_history_row(exp, row_key=str(i))
