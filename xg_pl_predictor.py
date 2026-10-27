"""
Premier League Match Predictor — v6
=====================================
Features vs v5:
  - Rolling xG for / against (Understat, all seasons) alongside goals, SOT and market signals
  - Prediction builds features through the same pipeline as training (no train/predict drift)
  - Prior-season weight now down-weights last season's games *within* a team's form window,
    instead of rescaling whole seasons (which changed what a feature value meant season to season)
  - Team list limited to the current season (relegated sides dropped)
  - Test metrics include log loss and the bookmaker-favourite baseline on the same matches

The Streamlit app (app.py) imports PLPredictor from here; run this file directly for a CLI demo.
"""

import gc
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Auto-download helpers
# ---------------------------------------------------------------------------
_FD_BASE = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
_SEASON_CODES = {
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
    "2026-27": "2627",
}

# Understat serves match-level xG for every season; keyed by the season's start year
_US_BASE = "https://understat.com/getLeagueData/EPL/{year}"
_US_TEAM_NAMES = {  # Understat name -> football-data.co.uk name
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
}


def _download_season(season: str) -> pd.DataFrame | None:
    code = _SEASON_CODES.get(season)
    if not code:
        return None
    try:
        r = requests.get(_FD_BASE.format(code=code), timeout=10)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text))
    except Exception:
        return None


def _download_xg(season: str) -> pd.DataFrame | None:
    """Played matches for one season from Understat: home_team, away_team, home_xg, away_xg."""
    year = season[:4]
    try:
        r = requests.get(
            _US_BASE.format(year=year),
            headers={
                "User-Agent": "Mozilla/5.0",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": f"https://understat.com/league/EPL/{year}",
            },
            timeout=15,
        )
        r.raise_for_status()
        games = [g for g in r.json()["dates"] if g.get("isResult")]
    except Exception:
        return None
    if not games:
        return None
    return pd.DataFrame({
        "home_team": [_US_TEAM_NAMES.get(g["h"]["title"], g["h"]["title"]) for g in games],
        "away_team": [_US_TEAM_NAMES.get(g["a"]["title"], g["a"]["title"]) for g in games],
        "home_xg": [float(g["xG"]["h"]) for g in games],
        "away_xg": [float(g["xG"]["a"]) for g in games],
    })


# Per-team-per-match stats rolled into form features
_STATS = ["pts", "gf", "ga", "sot", "mkt_win", "ah", "xgf", "xga"]
_DEFAULTS = dict(form=5.0, gf=1.5, ga=1.5, strength=50.0, sot=4.0, mkt=1/3, ah=0.0, xgf=1.4, xga=1.4)


class PLPredictor:
    SEASON_FILES = {
        "2022-23": ["PL_2223.csv", "PL 22-23.csv"],
        "2023-24": ["PL_2324.csv", "PL 23-24.csv"],
        "2024-25": ["PL_2425.csv", "PL 24-25.csv"],
        "2025-26": ["PL_2526.csv", "PL 25-26.csv"],
        "2026-27": ["PL_2627.csv", "PL 26-27.csv"],
    }
    # Understat snapshots, used only when the live xG download fails
    XG_FILES = {season: f"PL xG {season[2:4]}-{season[5:]}.csv" for season in SEASON_FILES}

    def __init__(self, short_window: int = 5, long_window: int = 15, prior_season_weight: float = 1.0):
        self.short_window = short_window
        self.long_window = long_window
        self.prior_season_weight = prior_season_weight
        self.label_encoder = LabelEncoder()
        self.model = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            eval_metric="mlogloss",
            tree_method="hist",
            early_stopping_rounds=50,
        )
        self.match_data: pd.DataFrame | None = None
        self._feature_columns: list[str] = []
        self.metrics: dict = {}
        self.class_report: str | None = None
        self.feature_importance_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # 1. Data loading
    # ------------------------------------------------------------------
    def get_football_data(
        self,
        include_historical: bool = False,
        fetch_season=_download_season,
        fetch_xg=_download_xg,
    ) -> tuple[bool, list[str]]:
        """fetch_season / fetch_xg can be swapped for cached versions (the Streamlit app does)."""
        seasons = ["2024-25", "2025-26", "2026-27"]
        if include_historical:
            seasons = ["2022-23", "2023-24"] + seasons

        all_matches, log = [], []
        for season in seasons:
            raw = fetch_season(season)
            source = "football-data.co.uk"
            if raw is None:
                raw = self._read_local(self.SEASON_FILES.get(season, []))
                source = "local file"
            if raw is None:
                log.append(f"⚠️ {season} — not found online or locally")
                continue
            try:
                cleaned = self._clean_season(raw, season)
            except Exception as e:
                log.append(f"❌ {season} — {e}")
                continue

            xg = fetch_xg(season)
            xg_source = "Understat"
            if xg is None:
                xg = self._read_local([self.XG_FILES[season]])
                xg_source = "local Understat snapshot"
            if xg is not None:
                # Each fixture is played once per season, so (home, away) is a unique key
                xg = xg.drop_duplicates(["home_team", "away_team"], keep="last")
                cleaned = cleaned.merge(xg, on=["home_team", "away_team"], how="left")
            else:
                cleaned["home_xg"] = cleaned["away_xg"] = np.nan
                xg_source = "unavailable"
            n_xg = cleaned["home_xg"].notna().sum()

            all_matches.append(cleaned)
            log.append(f"✅ {season} — {len(cleaned)} matches ({source}), xG for {n_xg} ({xg_source})")

        if not all_matches:
            return False, log

        self.match_data = pd.concat(all_matches, ignore_index=True)
        self.match_data.sort_values("date", inplace=True, kind="stable")
        self.match_data.reset_index(drop=True, inplace=True)
        log.append(f"📊 Total: {len(self.match_data)} matches")
        return True, log

    @staticmethod
    def _read_local(candidates: list[str]) -> pd.DataFrame | None:
        path = next((_HERE / p for p in candidates if (_HERE / p).exists()), None)
        if path is None:
            return None
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return None

    def _clean_season(self, data: pd.DataFrame, season: str) -> pd.DataFrame:
        rows = []
        for _, row in data.iterrows():
            if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
                continue
            try:
                hg, ag = int(row["FTHG"]), int(row["FTAG"])
            except (ValueError, TypeError):
                continue

            result = "H" if hg > ag else ("A" if ag > hg else "D")
            date = pd.to_datetime(row.get("Date", pd.NaT), errors="coerce", dayfirst=True)

            try:
                hst = int(float(row.get("HST") or 4))
                ast_val = int(float(row.get("AST") or 4))
            except (ValueError, TypeError):
                hst, ast_val = 4, 4

            try:
                raw_h = 1.0 / float(row.get("AvgH") or row.get("B365H") or 3.0)
                raw_d = 1.0 / float(row.get("AvgD") or row.get("B365D") or 3.3)
                raw_a = 1.0 / float(row.get("AvgA") or row.get("B365A") or 3.0)
                total = raw_h + raw_d + raw_a
                imp_h, imp_d, imp_a = raw_h / total, raw_d / total, raw_a / total
            except (ValueError, TypeError, ZeroDivisionError):
                imp_h, imp_d, imp_a = 1/3, 1/3, 1/3

            try:
                ah = float(row.get("AHh") or 0)
            except (ValueError, TypeError):
                ah = 0.0

            rows.append(dict(
                season=season,
                date=date,
                home_team=row.get("HomeTeam", ""),
                away_team=row.get("AwayTeam", ""),
                home_goals=hg,
                away_goals=ag,
                result=result,
                hst=hst,
                ast=ast_val,
                implied_h=imp_h,
                implied_d=imp_d,
                implied_a=imp_a,
                ah_line=ah,
            ))

        return pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    def _rolling_team_stats(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """One row per (match, team) with that team's form over its previous `window` games.

        Games from an earlier season than the match being described count prior_season_weight
        as much as same-season games. Because games are date-sorted, the same-season games are
        always the most recent part of the window, so they can be rolled separately and the
        earlier-season part recovered as (whole window − same-season part).
        """
        home = df[["match_id", "date", "season", "home_team", "h_pts", "home_goals", "away_goals",
                   "hst", "implied_h", "ah_line", "home_xg", "away_xg"]].copy()
        away = df[["match_id", "date", "season", "away_team", "a_pts", "away_goals", "home_goals",
                   "ast", "implied_a", "ah_line", "away_xg", "home_xg"]].copy()
        home.columns = away.columns = ["match_id", "date", "season", "team"] + _STATS
        away["ah"] = -away["ah"]  # AH line from the away team's perspective
        long = pd.concat([home, away]).sort_values(["team", "date", "match_id"], kind="stable").reset_index(drop=True)

        def roll(keys):
            g = long.groupby(keys)[_STATS]
            sums = g.transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum()).fillna(0)
            counts = g.transform(lambda x: x.shift(1).rolling(window, min_periods=1).count()).fillna(0)
            return sums, counts

        all_sum, all_cnt = roll("team")
        same_sum, same_cnt = roll(["team", "season"])
        w = self.prior_season_weight
        num = same_sum + w * (all_sum - same_sum)
        den = same_cnt + w * (all_cnt - same_cnt)
        avg = num / den.where(den > 0)

        out = long[["match_id", "team"]].copy()
        out["form"] = num["pts"].where(den["pts"] > 0)
        out["strength"] = (avg["pts"] * 20 + 20).clip(10, 90)
        for col in ["gf", "ga", "sot", "ah", "xgf", "xga"]:
            out[col] = avg[col]
        out["mkt"] = avg["mkt_win"]
        return out

    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy().sort_values("date", kind="stable").reset_index(drop=True)
        df["match_id"] = np.arange(len(df))
        df["h_pts"] = df["result"].map({"H": 3, "D": 1, "A": 0})
        df["a_pts"] = df["result"].map({"A": 3, "D": 1, "H": 0})

        for prefix, window in [("short", self.short_window), ("long", self.long_window)]:
            stats = self._rolling_team_stats(df, window)
            for side, team_col in [("h", "home_team"), ("a", "away_team")]:
                merged = df[["match_id", team_col]].merge(
                    stats, left_on=["match_id", team_col], right_on=["match_id", "team"], how="left"
                )
                for stat, default in _DEFAULTS.items():
                    df[f"{side}_{prefix}_{stat}"] = merged[stat].fillna(default).values
            del stats
        gc.collect()

        df["home_advantage"] = 1
        df.drop(columns=["h_pts", "a_pts"], inplace=True)

        for window in ("short", "long"):
            for stat in ("strength", "form", "gf", "sot", "mkt", "xgf", "xga"):
                df[f"{stat}_diff_{window}"] = df[f"h_{window}_{stat}"] - df[f"a_{window}_{stat}"]
        return df

    # ------------------------------------------------------------------
    # 3. Training
    # ------------------------------------------------------------------
    def _feature_cols(self) -> list[str]:
        per_team = [f"{side}_{window}_{stat}" for window in ("short", "long") for side in ("h", "a")
                    for stat in ("strength", "form", "gf", "ga", "sot", "mkt", "ah", "xgf", "xga")]
        diffs = [f"{stat}_diff_{window}" for stat in ("strength", "form", "gf", "sot", "mkt", "xgf", "xga")
                 for window in ("short", "long")]
        return per_team + diffs + ["home_advantage"]

    @staticmethod
    def _time_split(X, y, train_frac=0.70, val_frac=0.15):
        n = len(X)
        t, v = int(n * train_frac), int(n * (train_frac + val_frac))
        return X.iloc[:t], y[:t], X.iloc[t:v], y[t:v], X.iloc[v:], y[v:]

    def train_model(self) -> tuple[bool, list[str]]:
        if self.match_data is None:
            return False, ["No data loaded."]
        log = []

        featured = self._build_features(self.match_data)
        featured = featured.iloc[self.long_window:].reset_index(drop=True)

        self._feature_columns = self._feature_cols()
        X = featured[self._feature_columns].copy()
        y = self.label_encoder.fit_transform(featured["result"])
        log.append(f"Classes: {list(self.label_encoder.classes_)}")
        log.append(f"Dataset size: {len(X)} matches, {len(self._feature_columns)} features")

        X_train, y_train, X_val, y_val, X_test, y_test = self._time_split(X, y)
        log.append(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        sample_weights = np.exp(np.linspace(0, 1, len(X_train)))
        self.model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=False)

        proba = self.model.predict_proba(X_test)
        preds = proba.argmax(axis=1)

        # Bookmaker baseline on the same matches: pre-match implied probabilities, in label order
        test_rows = featured.iloc[len(X_train) + len(X_val):]
        implied = {"H": "implied_h", "D": "implied_d", "A": "implied_a"}
        market = test_rows[[implied[c] for c in self.label_encoder.classes_]].to_numpy()
        home_idx = list(self.label_encoder.classes_).index("H")
        labels = np.arange(len(self.label_encoder.classes_))

        self.metrics = dict(
            test_matches=len(y_test),
            test_from=test_rows["date"].min(),
            test_to=test_rows["date"].max(),
            accuracy=accuracy_score(y_test, preds),
            log_loss=log_loss(y_test, proba, labels=labels),
            market_accuracy=accuracy_score(y_test, market.argmax(axis=1)),
            market_log_loss=log_loss(y_test, market, labels=labels),
            home_win_rate=float(np.mean(y_test == home_idx)),
        )
        self.class_report = classification_report(
            y_test, preds, target_names=self.label_encoder.classes_, zero_division=0
        )
        self.feature_importance_df = pd.DataFrame({
            "Feature": self._feature_columns,
            "Importance": self.model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        m = self.metrics
        log.append(
            f"✅ Test ({m['test_from']:%d %b %Y} – {m['test_to']:%d %b %Y}): "
            f"accuracy {m['accuracy']:.1%}, log loss {m['log_loss']:.3f}"
        )
        log.append(
            f"   Bookmaker favourite on same matches: accuracy {m['market_accuracy']:.1%}, "
            f"log loss {m['market_log_loss']:.3f} · home wins {m['home_win_rate']:.1%}"
        )
        del featured
        gc.collect()
        return True, log

    def walk_forward(self, start_season: str = "2024-25") -> pd.DataFrame:
        """Retrain before each half-season from start_season on and score it unseen.

        Steadier than the single train_model() split, which only covers the most recent ~240
        matches. Each block is scored against the bookmaker favourite on the same matches.
        """
        featured = self._build_features(self.match_data)
        featured = featured.iloc[self.long_window:].reset_index(drop=True)
        featured["block"] = featured["season"] + np.where(featured["date"].dt.month >= 7, " H1", " H2")
        cols = self._feature_cols()
        encoder = LabelEncoder()
        y = encoder.fit_transform(featured["result"])
        implied = {"H": "implied_h", "D": "implied_d", "A": "implied_a"}
        market_cols = [implied[c] for c in encoder.classes_]
        home_idx = list(encoder.classes_).index("H")
        labels = np.arange(len(encoder.classes_))

        def score(name, y_true, proba, market):
            return dict(
                block=name,
                matches=len(y_true),
                accuracy=accuracy_score(y_true, proba.argmax(axis=1)),
                log_loss=log_loss(y_true, proba, labels=labels),
                market_accuracy=accuracy_score(y_true, market.argmax(axis=1)),
                market_log_loss=log_loss(y_true, market, labels=labels),
                home_win_rate=float(np.mean(y_true == home_idx)),
            )

        rows, all_y, all_p, all_m = [], [], [], []
        for block in sorted(b for b in featured["block"].unique() if b >= start_season):
            test = featured.index[featured["block"] == block]
            train = featured.index[featured["date"] < featured.loc[test, "date"].min()]
            cut = int(len(train) * 0.85)  # last 15% of history for early stopping
            fit, val = train[:cut], train[cut:]

            model = XGBClassifier(**self.model.get_params())
            model.fit(
                featured.loc[fit, cols], y[fit], sample_weight=np.exp(np.linspace(0, 1, len(fit))),
                eval_set=[(featured.loc[val, cols], y[val])], verbose=False,
            )
            proba = model.predict_proba(featured.loc[test, cols])
            market = featured.loc[test, market_cols].to_numpy()
            rows.append(score(block, y[test], proba, market))
            all_y.append(y[test]); all_p.append(proba); all_m.append(market)

        if rows:
            rows.append(score("Overall", np.concatenate(all_y), np.vstack(all_p), np.vstack(all_m)))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 4. Prediction
    # ------------------------------------------------------------------
    def known_teams(self) -> list[str]:
        """Teams in the latest loaded season (relegated sides from earlier seasons excluded)."""
        if self.match_data is None:
            return []
        current = self.match_data[self.match_data["season"] == self.match_data["season"].max()]
        return sorted(set(current["home_team"]) | set(current["away_team"]))

    def _validate_team(self, name: str) -> str:
        teams = self.known_teams()
        if name in teams:
            return name
        suggestions = [t for t in teams if name.lower() in t.lower() or t.lower() in name.lower()]
        msg = f"Unknown team '{name}'."
        if suggestions:
            msg += f" Did you mean one of: {suggestions}?"
        else:
            msg += f" Current-season teams: {teams}"
        raise ValueError(msg)

    def predict_match(self, home_team: str, away_team: str) -> dict:
        home_team = self._validate_team(home_team)
        away_team = self._validate_team(away_team)

        # Append the fixture as an unplayed match so its features come from the training pipeline
        fixture = pd.DataFrame([dict(
            season=self.match_data["season"].max(),
            date=self.match_data["date"].max() + pd.Timedelta(days=1),
            home_team=home_team,
            away_team=away_team,
        )])
        featured = self._build_features(pd.concat([self.match_data, fixture], ignore_index=True))
        row = featured.iloc[-1]
        feat = featured[self._feature_columns].iloc[[-1]]

        proba = self.model.predict_proba(feat)[0]
        classes = self.label_encoder.inverse_transform(np.arange(len(proba)))
        pred_lbl = classes[proba.argmax()]

        def team_stats(side):
            return {
                window: dict(
                    strength=row[f"{side}_{window}_strength"],
                    form=row[f"{side}_{window}_form"],
                    goals_for=row[f"{side}_{window}_gf"],
                    goals_against=row[f"{side}_{window}_ga"],
                    xg_for=row[f"{side}_{window}_xgf"],
                    xg_against=row[f"{side}_{window}_xga"],
                )
                for window in ("short", "long")
            }

        return {
            "pred_lbl": pred_lbl,
            "probas": dict(zip(classes, proba)),
            "features": feat.iloc[0],
            "home_stats": team_stats("h"),
            "away_stats": team_stats("a"),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import plotly.express as px

    print("Premier League Predictor — v6")
    print("=" * 55)

    predictor = PLPredictor(short_window=5, long_window=15, prior_season_weight=1.0)

    ok, log = predictor.get_football_data(include_historical=True)
    print("\n".join(log))
    if not ok:
        return

    ok, log = predictor.train_model()
    print("\n".join(log))
    if not ok:
        return
    print("\nClassification report:")
    print(predictor.class_report)
    if predictor.model.best_iteration:
        print(f"Best XGBoost iteration: {predictor.model.best_iteration}")

    print("\nWalk-forward evaluation (retrained before each half-season, scored unseen):")
    wf = predictor.walk_forward()
    print(wf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fi = predictor.feature_importance_df
    fig = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        title="Feature Importance (XGBoost v6)",
        text=fi["Importance"].round(3),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white")
    fig.show()

    print("\nCurrent-season teams:", predictor.known_teams())

    matches_to_predict = [
        ("Tottenham", "Arsenal"),
        ("Liverpool", "Man City"),
        ("Chelsea", "Man United"),
    ]

    for home, away in matches_to_predict:
        try:
            result = predictor.predict_match(home, away)
        except ValueError as e:
            print(f"[skip] {e}")
            continue

        print(f"\nPREDICTING: {home} (H)  vs  {away} (A)")
        print("=" * 55)
        print("\nFeatures fed to model:")
        print(result["features"].to_frame("value").to_string())

        labels = {"H": f"{home} Win", "D": "Draw", "A": f"{away} Win"}
        print(f"\nPREDICTION: {labels[result['pred_lbl']]}")
        print("Probabilities:")
        for lbl, p in sorted(result["probas"].items(), key=lambda x: -x[1]):
            print(f"  {labels[lbl]:<25} {p:.1%}")

        prob_df = pd.DataFrame({
            "Outcome": [labels[l] for l in result["probas"]],
            "Probability": list(result["probas"].values()),
        })
        fig = px.pie(prob_df, names="Outcome", values="Probability", title=f"{home} vs {away}")
        fig.show()


if __name__ == "__main__":
    main()
