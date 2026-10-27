"""
Premier League Match Predictor — Streamlit App
Model lives in xg_pl_predictor.py (v6)
Run with: streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from xg_pl_predictor import PLPredictor, _download_season, _download_xg

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PL Match Predictor",
    page_icon="⚽",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached downloads (football-data.co.uk results/odds + Understat xG)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_season(season: str) -> pd.DataFrame | None:
    return _download_season(season)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_xg(season: str) -> pd.DataFrame | None:
    return _download_xg(season)


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

# Cache the trained model so it only retrains when settings change
@st.cache_resource(show_spinner=False)
def load_and_train(include_historical: bool, short_window: int, long_window: int, prior_weight: float):
    p = PLPredictor(short_window=short_window, long_window=long_window, prior_season_weight=prior_weight)
    ok, data_log = p.get_football_data(
        include_historical=include_historical, fetch_season=cached_season, fetch_xg=cached_xg
    )
    if not ok:
        return None, data_log, []
    ok2, train_log = p.train_model()
    return (p if ok2 else None), data_log, train_log


# --- Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg", width=120)
st.sidebar.title("⚙️ Settings")

include_hist = st.sidebar.toggle("Include 22/23 & 23/24 seasons", value=True)
short_w = st.sidebar.slider("Short form window (games)", 3, 10, 5)
long_w  = st.sidebar.slider("Long form window (games)", 10, 20, 15)
prior_w = st.sidebar.slider(
    "Prior season weight", 0.1, 1.0, 1.0, step=0.05,
    help="How much a previous season's games count in a team's form window, relative to this season's",
)

st.sidebar.divider()
st.sidebar.caption("Results & odds from football-data.co.uk · xG from Understat · Local CSVs used as fallback")
if st.sidebar.button("🔄 Refresh data", help="Clears the download cache and re-fetches all seasons"):
    cached_season.clear()
    cached_xg.clear()
    load_and_train.clear()
    st.rerun()

# --- Header ---
st.title("⚽ Premier League Match Predictor")
st.caption("XGBoost model · Dual form windows · Rolling xG · Season-weighted lookback")

# --- Load & train ---
with st.spinner("Loading data and training model…"):
    predictor, data_log, train_log = load_and_train(include_hist, short_w, long_w, prior_w)

with st.expander("📋 Training log", expanded=False):
    for line in data_log + train_log:
        st.text(line)

if predictor is None:
    st.error("Model failed to load. Check that your CSV files are in the right place.")
    st.stop()

# --- Model metrics ---
m = predictor.metrics
st.caption(
    f"Test set: last {m['test_matches']} matches "
    f"({m['test_from']:%d %b %Y} – {m['test_to']:%d %b %Y}) · home win rate {m['home_win_rate']:.1%}"
)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Test Accuracy", f"{m['accuracy']:.1%}")
with col2:
    st.metric("Bookmaker Favourite Accuracy", f"{m['market_accuracy']:.1%}", help="Same test matches")
with col3:
    st.metric(
        "Log Loss", f"{m['log_loss']:.3f}",
        delta=f"{m['log_loss'] - m['market_log_loss']:+.3f} vs bookmaker", delta_color="inverse",
        help="Lower is better; scores the probabilities, not just the pick",
    )
with col4:
    if predictor.model.best_iteration:
        st.metric("Best XGB Iteration", predictor.model.best_iteration)

with st.expander("📊 Full classification report"):
    st.code(predictor.class_report)


@st.cache_data(show_spinner=False)
def walk_forward_table(include_historical: bool, short_window: int, long_window: int, prior_weight: float):
    p, _, _ = load_and_train(include_historical, short_window, long_window, prior_weight)
    return p.walk_forward()


with st.expander("📆 Walk-forward accuracy (by half-season)"):
    st.caption("Model retrained before each half-season and scored on it unseen, next to the bookmaker favourite")
    with st.spinner("Running walk-forward evaluation…"):
        wf = walk_forward_table(include_hist, short_w, long_w, prior_w)
    st.dataframe(
        wf.rename(columns={
            "block": "Period", "matches": "Matches", "accuracy": "Model Acc", "log_loss": "Model Log Loss",
            "market_accuracy": "Bookmaker Acc", "market_log_loss": "Bookmaker Log Loss", "home_win_rate": "Home Win %",
        }).style.format({
            "Model Acc": "{:.1%}", "Bookmaker Acc": "{:.1%}", "Home Win %": "{:.1%}",
            "Model Log Loss": "{:.3f}", "Bookmaker Log Loss": "{:.3f}",
        }),
        width="stretch", hide_index=True,
    )

# --- Feature importance chart ---
with st.expander("🔍 Feature importance"):
    fi = predictor.feature_importance_df
    fig_fi = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        text=fi["Importance"].round(3), template="plotly_white",
        title="Feature Importance (XGBoost v6)",
    )
    fig_fi.update_traces(textposition="outside")
    fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=max(550, 18 * len(fi)))
    st.plotly_chart(fig_fi, width="stretch")

st.divider()

# --- Match prediction ---
st.subheader("🔮 Predict a Match")

teams = predictor.known_teams()
col_h, col_vs, col_a = st.columns([5, 1, 5])
with col_h:
    home_team = st.selectbox("🏠 Home Team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
with col_vs:
    st.markdown("<br><h3 style='text-align:center'>vs</h3>", unsafe_allow_html=True)
with col_a:
    away_team = st.selectbox("✈️ Away Team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)

if home_team == away_team:
    st.warning("Please select two different teams.")
elif st.button("⚡ Predict", type="primary", width="stretch"):
    result = predictor.predict_match(home_team, away_team)
    probas  = result["probas"]
    pred    = result["pred_lbl"]

    labels = {"H": f"{home_team} Win", "D": "Draw", "A": f"{away_team} Win"}
    pred_label = labels[pred]

    # Outcome banner
    colour = "#2ecc71" if pred == "H" else ("#e67e22" if pred == "D" else "#e74c3c")
    st.markdown(
        f"""<div style='background:{colour};padding:18px;border-radius:10px;text-align:center;'>
        <h2 style='color:white;margin:0'>🏆 {pred_label}</h2></div>""",
        unsafe_allow_html=True,
    )
    st.write("")

    # Probability columns
    r1, r2, r3 = st.columns(3)
    for col, outcome_key, label in [(r1, "H", f"{home_team} Win"), (r2, "D", "Draw"), (r3, "A", f"{away_team} Win")]:
        p = probas.get(outcome_key, 0)
        col.metric(label, f"{p:.1%}")

    # Pie chart
    prob_df = pd.DataFrame({
        "Outcome": [labels[k] for k in probas],
        "Probability": list(probas.values()),
    })
    fig_pie = px.pie(
        prob_df, names="Outcome", values="Probability",
        title=f"{home_team} vs {away_team}",
        color_discrete_sequence=["#2ecc71", "#e67e22", "#e74c3c"],
    )
    fig_pie.update_traces(textinfo="label+percent")
    st.plotly_chart(fig_pie, width="stretch")

    # Team stats comparison
    st.divider()
    st.subheader("📈 Recent Form Comparison")
    hs, as_ = result["home_stats"]["short"], result["away_stats"]["short"]
    hl, al  = result["home_stats"]["long"],  result["away_stats"]["long"]

    def column(s):
        return [round(s["strength"], 1), round(s["form"], 1), round(s["goals_for"], 2), round(s["goals_against"], 2),
                round(s["xg_for"], 2), round(s["xg_against"], 2)]

    stats_df = pd.DataFrame({
        "Stat": ["Strength", "Form (pts)", "Goals For / game", "Goals Against / game", "xG For / game", "xG Against / game"],
        f"{home_team} (short)": column(hs),
        f"{away_team} (short)": column(as_),
        f"{home_team} (long)":  column(hl),
        f"{away_team} (long)":  column(al),
    })
    st.dataframe(stats_df, width="stretch", hide_index=True)
