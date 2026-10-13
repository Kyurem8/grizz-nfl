"""Production refresh: established score-only margin and EPA-blended totals.
No live betting prices; forecast probabilities intentionally withheld.
"""
import argparse,json,csv,sys
from pathlib import Path
from datetime import datetime,timezone
import requests
import pandas as pd
from grizz.core import load_games,fit_ratings,predict
from grizz.advanced import load_or_refresh_features,enhanced_ratings
from run import SOURCE
ROOT=Path(__file__).resolve().parent

def refresh(offline=False,games_count=1502):
    today=datetime.now(timezone.utc).date().isoformat()
    data=ROOT/'data';out=ROOT/'output';data.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    games_path=data/'games.csv'
    if not offline:
        r=requests.get(SOURCE,timeout=90,headers={'User-Agent':'GrizzRatings/0.4'});r.raise_for_status()
        if not r.content.startswith(b'game_id,'):raise ValueError('Unexpected nflverse games schema')
        games_path.write_bytes(r.content)
    if not games_path.exists():raise FileNotFoundError('Missing data/games.csv; run online once')
    history,allgames=load_games(games_path,today,games_count)
    features=load_or_refresh_features(data/'game_epa_features.csv',history)
    if not set(history.game_id.astype(str)).issubset(set(features.game_id.astype(str))):raise ValueError('Incomplete EPA coverage')
    score,score_meta=fit_ratings(history)
    advanced,advanced_meta=enhanced_ratings(history,features,blend=.32)
    manual={}
    with (data/'adjustments.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            if row.get('team'):manual[row['team']]=float(row.get('points_adjustment') or 0)
    # A clear public-facing ratings table: 32% EPA is descriptive, NOT the selected margin model.
    ratings=advanced.round(4).to_dict('records')
    futures=allgames.loc[(allgames.date>=pd.Timestamp(today))&allgames.home_score.isna()&allgames.away_score.isna()].sort_values(['date','game_id']).head(80)
    picks=[]
    for g in futures.itertuples():
        sp=predict(g.home_team,g.away_team,score,score_meta,manual)
        tp=predict(g.home_team,g.away_team,advanced,advanced_meta,manual)
        margin=sp['home_margin'];total=tp['total']
        picks.append({'date':str(g.date.date()),'matchup':f'{g.away_team} @ {g.home_team}', 'away':g.away_team,'home':g.home_team,'away_points':round((total-margin)/2,1),'home_points':round((total+margin)/2,1),'home_margin':round(margin,1),'total':round(total,1)})
    # JSON write only when ALL steps succeed; no partial output published.
    payload={'version':'0.4.0','generated_utc':datetime.now(timezone.utc).isoformat(),'training_games':len(history),'last_game':str(history.date.max().date()),'mean_points':score_meta['baseline_points_per_team'],'home_advantage':score_meta['home_field_points'],'spread_model':'Score-only (V0.1 benchmark)','total_model':'EPA blend 32% (V0.2 benchmark)','ratings_model':'EPA blend 32% — descriptive power ratings','test_reference':{'games':137,'v01_margin_mae':10.399,'v02_total_mae':10.826,'v01_winner_brier':0.2369,'note':'Historical single holdout; do not interpret as updated backtest'},'ratings':ratings,'picks':picks,'manual_adjustments':manual,'market_status':'Not connected','probabilities_status':'Not calibrated / not published'}
    temp=out/'current.tmp';temp.write_text(json.dumps(payload,indent=2),encoding='utf8');temp.replace(out/'current.json')
    print(f'Grizz V0.4 refreshed: {len(history)} verified games; {len(picks)} upcoming matchups; last {payload["last_game"]}')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--offline',action='store_true');a.add_argument('--games',type=int,default=1502);args=a.parse_args()
    try:refresh(args.offline,args.games)
    except Exception as exc:
        print(f'REFRESH FAILED: {exc}',file=sys.stderr);raise
