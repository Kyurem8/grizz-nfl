"""One-command Grizz Ratings refresh. Use --offline for existing cached games.csv."""
import argparse, csv, json, sys
from datetime import datetime, timedelta
from pathlib import Path
import requests
from grizz.core import load_games,fit_ratings,predict,backtest

ROOT=Path(__file__).resolve().parent
SOURCE="https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--offline",action="store_true",help="Use existing data/games.csv")
    p.add_argument("--cutoff",default=datetime.now().date().isoformat(),help="YYYY-MM-DD; strictly earlier games only")
    p.add_argument("--games",type=int,default=1502)
    p.add_argument("--backtest",type=int,default=128,help="Number of last training games to walk forward evaluate")
    a=p.parse_args(); path=ROOT/'data'/'games.csv';path.parent.mkdir(exist_ok=True)
    if not a.offline:
        print('Fetching source NFL results from nflverse...')
        try:
            response=requests.get(SOURCE,timeout=50,headers={"User-Agent":"GrizzRatings/0.1"})
            response.raise_for_status()
            if not response.content.startswith(b'game_id,'):raise ValueError('Unexpected dataset format')
            path.write_bytes(response.content)
        except Exception as e:
            if not path.exists():raise RuntimeError(f'Download failed. Connection required for first run: {e}')
            print('WARNING: Download failed; using cached file:',e)
    games,allgames=load_games(path,a.cutoff,a.games)
    print(f'Verified training set: {len(games)} completed games from {games.date.min().date()} through {games.date.max().date()}')
    ratings,meta=fit_ratings(games)
    ratings.to_csv(ROOT/'output'/'grizz_ratings.csv',index=False,float_format='%.4f')
    metrics,replay=backtest(games,holdout=min(a.backtest,len(games)-800)) if a.backtest>0 else ({"games_tested":0},None)
    if replay is not None and len(replay):replay.to_csv(ROOT/'output'/'backtest_predictions.csv',index=False)
    adjustments={}
    ap=ROOT/'data'/'adjustments.csv'
    if ap.exists():
        with ap.open(newline='') as f:
            for r in csv.DictReader(f):
                if r['team'] in ratings.team.values:adjustments[r['team']]=float(r.get('points_adjustment') or 0)
    from pandas import Timestamp
    upcoming=allgames.loc[(allgames.date >= Timestamp(a.cutoff)) & allgames.home_score.isna() & allgames.away_score.isna()].sort_values(['date','game_id']).head(48)
    picks=[]
    for g in upcoming.itertuples():
        projection=predict(g.home_team,g.away_team,ratings,meta,adjustments)
        picks.append({"date":str(g.date.date()),"game":f"{g.away_team} @ {g.home_team}","home":g.home_team,"away":g.away_team,**projection,"status":"Projection only — not calibrated fair odds"})
    result={"generated_at":datetime.now().isoformat(timespec='seconds'),"cutoff":a.cutoff,"model":"Opponent-adjusted points ridge, rolling exponentially weighted training","meta":meta,"backtest":metrics,"teams":ratings.round(3).to_dict(orient='records'),"games":picks,"manual_adjustments":adjustments,"markets_status":"Not connected; no live Kalshi/Polymarket/Novig lines"}
    (ROOT/'output'/'ratings.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    template=(ROOT/'web_template.html').read_text(encoding='utf8')
    (ROOT/'output'/'index.html').write_text(template.replace('/*GRIZZ_DATA*/',json.dumps(result)),encoding='utf8')
    print('\nTOP 10 Grizz ratings:')
    print(ratings[['overall_rank','team','overall','offense_points','defense_allowed']].head(10).round(2).to_string(index=False))
    print('\nWalk-forward backtest:',metrics)
    print('\nCreated output/index.html, grizz_ratings.csv, ratings.json, backtest_predictions.csv')
    print('Open output/index.html in your browser. Training data is in data/games.csv.')

if __name__=='__main__':
    try:main()
    except Exception as e:print(f'ERROR: {e}',file=sys.stderr);sys.exit(1)
