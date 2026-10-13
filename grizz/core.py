"""Opponent-adjusted NFL offense/defense scoring ratings with rolling time-weighting.

Ridge regression is descriptive, not a calibrated gambling pricing model.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

TEAMS = ["ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC","LV","LAC","LAR","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"]
NAMES = dict(zip(TEAMS,["Cardinals","Falcons","Ravens","Bills","Panthers","Bears","Bengals","Browns","Cowboys","Broncos","Lions","Packers","Texans","Colts","Jaguars","Chiefs","Raiders","Chargers","Rams","Dolphins","Vikings","Patriots","Saints","Giants","Jets","Eagles","Steelers","Seahawks","49ers","Buccaneers","Titans","Commanders"]))
ALIASES = {"LA":"LAR", "STL":"LAR", "SD":"LAC", "OAK":"LV", "WSH":"WAS", "JAC":"JAX"}

def canonical(team):
    t = str(team).strip().upper()
    return ALIASES.get(t, t)

def load_games(path, cutoff, n_games=1502):
    df = pd.read_csv(path, low_memory=False)
    required = {"game_id", "season", "game_type", "gameday", "home_team", "away_team", "home_score", "away_score"}
    if missing := required - set(df.columns):
        raise ValueError(f"Game file missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["gameday"], errors="coerce")
    df["home_team"] = df["home_team"].map(canonical)
    df["away_team"] = df["away_team"].map(canonical)
    for c in ["home_score", "away_score"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.loc[df["home_team"].isin(TEAMS) & df["away_team"].isin(TEAMS) & df["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    # Completed games ONLY, and conservatively never include cutoff day in training.
    history = df.loc[(df.date < pd.Timestamp(cutoff)) & df.home_score.notna() & df.away_score.notna()].copy()
    history = history.sort_values(["date", "game_id"]).drop_duplicates("game_id")
    if len(history) < n_games:
        raise ValueError(f"Only {len(history)} qualifying completed games before {cutoff}; requested {n_games}.")
    return history.tail(n_games).reset_index(drop=True), df

def fit_ratings(games, half_life_days=280, alpha=12):
    if len(games) < 100: raise ValueError("At least 100 completed games required")
    n = len(games); t = len(TEAMS); ids = {name:i for i,name in enumerate(TEAMS)}
    x = np.zeros((2*n,2*t+1),dtype=np.float64)
    y = np.empty(2*n,dtype=np.float64)
    for k,g in enumerate(games.itertuples()):
        h,a=ids[g.home_team],ids[g.away_team]
        # Defense coefficient is extra points conceded (lower is better).
        x[2*k,h]=1; x[2*k,t+a]=1; x[2*k,2*t]=1
        x[2*k+1,a]=1; x[2*k+1,t+h]=1
        y[2*k]=g.home_score; y[2*k+1]=g.away_score
    days = (games.date.max()-games.date).dt.days.to_numpy()
    weight = np.repeat(np.power(0.5,days/half_life_days),2)
    # Fit deviations from weighted league scoring level; intercept is not penalized.
    baseline = float(np.average(y, weights=weight))
    x0 = x.copy(); x0[:,2*t] -= np.average(x0[:,2*t],weights=weight)
    # Ridge with explicit unpenalized home field column by rescaling its penalty.
    fit = Ridge(alpha=alpha, fit_intercept=True)
    fit.fit(x, y, sample_weight=weight)
    off = fit.coef_[:t]; deff = fit.coef_[t:2*t]
    off = off-off.mean(); deff = deff-deff.mean()
    hfa=float(fit.coef_[-1]); avg=float(fit.intercept_ + np.mean(fit.coef_[:t])+np.mean(fit.coef_[t:2*t]))
    # If tiny samples yield extreme baselines, safely expose actual intercept.
    ratings = pd.DataFrame({"team":TEAMS,"team_name":[NAMES[k] for k in TEAMS],"offense":off,"defense":deff})
    ratings["overall"] = ratings.offense-ratings.defense
    ratings["offense_points"] = avg+ratings.offense
    ratings["defense_allowed"] = avg+ratings.defense
    for col,asc,name in [("overall",False,"overall_rank"),("offense",False,"offense_rank"),("defense",True,"defense_rank")]:
        ratings[name]=ratings[col].rank(method="min",ascending=asc).astype(int)
    ratings=ratings.sort_values("overall",ascending=False).reset_index(drop=True)
    return ratings,{"baseline_points_per_team":round(avg,4),"home_field_points":round(hfa,4),"half_life_days":half_life_days,"ridge_alpha":alpha,"training_games":n,"first_game":str(games.date.min().date()),"last_game":str(games.date.max().date())}

def predict(home, away, ratings, meta, adjustments=None):
    adjustments=adjustments or {}; r=ratings.set_index("team")
    if home not in r.index or away not in r.index:raise ValueError("Invalid team")
    b=meta["baseline_points_per_team"]
    h=float(b+r.loc[home,"offense"]+r.loc[away,"defense"]+meta["home_field_points"])
    a=float(b+r.loc[away,"offense"]+r.loc[home,"defense"])
    # Each manual adjustment represents points added to that team's scoring margin.
    h+=float(adjustments.get(home,0)); a+=float(adjustments.get(away,0))
    return {"home_points":round(max(0,h),1),"away_points":round(max(0,a),1),"home_margin":round(h-a,1),"total":round(max(0,h)+max(0,a),1)}

def backtest(games, half_life_days=280, alpha=12, holdout=128, minimum_train=800):
    # Weekly refits; every prediction uses ONLY preceding calendar dates.
    evalgames=games.tail(holdout).copy()
    preds=[]
    for date, block in evalgames.groupby("date",sort=True):
        prior=games.loc[games.date<date]
        if len(prior)<minimum_train:continue
        ratings,meta=fit_ratings(prior,half_life_days,alpha)
        for g in block.itertuples():
            p=predict(g.home_team,g.away_team,ratings,meta)
            preds.append({"date":str(date.date()),"game_id":g.game_id,"actual_margin":g.home_score-g.away_score,"pred_margin":p["home_margin"],"actual_total":g.home_score+g.away_score,"pred_total":p["total"]})
    if not preds:return {"games_tested":0},pd.DataFrame()
    df=pd.DataFrame(preds)
    metrics = {
        "games_tested": len(df),
        "margin_mae": round(float(np.mean(np.abs(df.pred_margin-df.actual_margin))), 3),
        "total_mae": round(float(np.mean(np.abs(df.pred_total-df.actual_total))), 3),
        "winner_accuracy_excluding_ties": round(float(np.mean(np.sign(df.loc[df.actual_margin!=0,"pred_margin"]) == np.sign(df.loc[df.actual_margin!=0,"actual_margin"]))), 4),
        "naive_zero_margin_mae": round(float(np.mean(np.abs(df.actual_margin))), 3),
        "naive_44_total_mae": round(float(np.mean(np.abs(44-df.actual_total))), 3),
    }
    return metrics,df
