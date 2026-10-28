# ⚽ Premier League Match Predictor

> An XGBoost-powered match prediction engine for the English Premier League — built with real historical data, dual form windows, rolling market signals, and a live Streamlit dashboard.

---

## 🧠 The Problem

Predicting football results is hard. Bookmakers employ teams of analysts and still get it wrong. Most hobbyist models use overly simple features — season-long averages that ignore momentum — or leak future data into training, making accuracy look better than it really is.

We built something better.

---

## 💡 Our Solution

A machine learning pipeline that:
- Trains on **5 seasons of Premier League data** (2022–2027, incl. the live 2026-27 season), **auto-downloaded** from football-data.co.uk — no manual CSV management
- Adds **match-level xG for every season** from Understat, rolled into xG-for / xG-against form features
- Uses **dual rolling form windows** (5-game short-term + 15-game long-term) to capture both momentum and underlying quality
- Incorporates **shots on target**, **market-implied win probabilities**, and **Asian handicap lines** — all derived from historical data, no external API needed
- Applies **season-aware weighting** — a tunable discount on last season's games inside each team's form window
- Builds prediction features through the **same pipeline as training**, so the app never feeds the model inputs it wasn't trained on
- Uses a **chronological train/val/test split** — no data leakage, no inflated accuracy
- Surfaces everything through a clean, interactive **Streamlit dashboard**

---

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/RKarnik10/-ML_PL_Project.git
cd ./-ML_PL_Project

# 2. Install dependencies
pip install streamlit xgboost scikit-learn plotly pandas requests

# 3. Run the app
streamlit run app.py
```

> Opens at `http://localhost:8501`. Data is fetched automatically on startup — no CSV files required.

---

## 🖥️ App Features

| Feature | Description |
|---|---|
| **Auto data refresh** | Fetches results/odds from football-data.co.uk and xG from Understat on load, cached for 1 hour |
| **🔄 Refresh button** | Force re-download after a new matchweek |
| **Team selector** | Pick any two current-season PL teams (relegated sides excluded) |
| **Live prediction** | Win/Draw/Loss probabilities with colour-coded result banner |
| **Form comparison table** | Short & long-term stats side by side for both teams |
| **Feature importance chart** | See which signals the model weights most heavily |
| **Configurable settings** | Tune form windows and season weights from the sidebar |
| **Model metrics** | Test accuracy and log loss shown next to the bookmaker-favourite baseline on the same matches |
| **Walk-forward table** | Per-half-season accuracy and log loss vs the bookmaker, retrained before each block |
| **Training log** | Transparent view of data loaded (incl. xG coverage) and model accuracy |

---

## 🏗️ Architecture

```
ML_PL_Project/
│
├── app.py                  # Streamlit frontend + UI logic (imports the model)
├── xg_pl_predictor.py      # Core model — PLPredictor (standalone, runnable directly)
│
├── PL 22-23.csv … PL 26-27.csv        # Local fallback for results/odds (auto-fetched if online)
├── PL xG 22-23.csv … PL xG 26-27.csv  # Local fallback for Understat xG
│
└── README.md
```

### Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost (`multi:softprob`) |
| Feature engineering | Pandas rolling windows, fully vectorised |
| Frontend | Streamlit + Plotly |
| Data | [football-data.co.uk](https://www.football-data.co.uk) results & odds, [Understat](https://understat.com/league/EPL) xG (auto-fetched via `requests`) |

---

## 📊 How the Model Works

### Features (51 total)

Each match is represented by rolling stats computed **strictly from prior matches** — no lookahead bias.

| Group | Features |
|---|---|
| Short form (last 5 games) | Strength, form pts, goals for/against, **xG for/against**, shots on target, market win prob, AH line — home & away |
| Long form (last 15 games) | Same as above, capturing underlying quality over a larger window |
| Difference features | Home minus away for each stat — gives the model relative signals |
| Context | Home advantage flag |

**xG features** (`xgf`, `xga`) are rolling averages of Understat expected goals — a less noisy read of chance quality than goals.

**Market features** (`mkt`, `ah`) are rolling averages of implied win probabilities and Asian handicap lines from historical match odds in the dataset — no external odds input needed at prediction time.

### Training

- **Algorithm**: XGBoost with `multi:softprob` objective
- **Split**: Chronological 70% train / 15% val / 15% test
- **Recency weighting**: Exponential sample weights (~2.7x for newest vs oldest matches)
- **Early stopping**: Prevents overfitting via validation loss monitoring
- **Prior season weight**: How much last season's games count inside a team's form window (default 1.0 = no discount; 0.4 tested worse in walk-forward evaluation)

---

## 📈 Results

Walk-forward evaluation: retrain before each half-season from 2024-25 onward and predict it unseen (840 matches).

| Model | Accuracy | Log loss |
|---|---|---|
| This model (v6, with xG) | 50.5% | 1.021 |
| Previous model (v5, no xG) | 48.8% | 1.035 |
| Bookmaker favourite (pre-match odds) | 51.4% | 0.995 |
| Always predict home win | 41.4% | — |

| Block | Matches | Model acc. | Model log loss | Bookmaker acc. | Bookmaker log loss | Home win rate |
|---|---|---|---|---|---|---|
| 2024-25 H1 | 188 | 47.9% | 1.028 | 50.0% | 0.990 | 40.4% |
| 2024-25 H2 | 192 | 55.7% | 1.002 | 58.3% | 0.952 | 41.1% |
| 2025-26 H1 | 186 | 53.8% | 0.973 | 54.3% | 0.971 | 46.8% |
| 2025-26 H2 | 194 | 47.4% | 1.073 | 44.8% | 1.058 | 38.7% |
| 2026-27 H1 | 80 | 43.8% | 1.037 | 47.5% | 1.020 | 38.8% |

> Accuracy swings a lot with the test window: the bookmakers themselves hit 55–61% in 2022–24 but only 45–48% across 2025-26 H2 and the start of 2026-27 (home wins down to ~39%, draws ~30%). The in-app test set is the most recent 15% of matches (currently 238 matches, 24 Jan – 25 Oct 2026: model 45.4% / 1.036 vs bookmaker 47.1% / 1.027), so it sits in that hard stretch — compare against the bookmaker number shown beside it, and prefer log loss over accuracy.

> Football is inherently unpredictable. The model beats the always-home-win baseline comfortably and gets close to the bookmakers on log loss, but doesn't beat them yet.

Reproduce with `python xg_pl_predictor.py` (prints the in-app test split and the walk-forward table).

---

## 🔮 What's Next

- **Home/away split stats** — separate rolling windows for home performance vs away performance
- **Scoreline prediction** — Poisson model to predict exact scores, not just outcomes
- **Node.js + FastAPI** — rebuild the frontend in Next.js with the Python model exposed as a REST API
- **Claude API integration** — AI-generated match previews based on team stats

---

## 👥 Team

| Name | Role |
|---|---|
| Your Name | ML Model, Data Pipeline |
| Teammate | Frontend, Streamlit UI |
| Teammate | Research, Pitch |

---

## 📂 Data Source

- **Results, shots and odds:** [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — free, reliable, updated after every matchweek.
- **Expected goals (xG):** [Understat](https://understat.com/league/EPL) — match-level xG for every season, matched to each fixture.

Snapshots of both are saved in the repo as an offline fallback.

---

## 📜 License

MIT — free to use, fork, and improve.
