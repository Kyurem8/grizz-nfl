"""EPA challenger layered onto Grizz scoring ratings.

All plays are filtered to *completed training game IDs*. Training/holdout splits
are done on game dates. EPA-to-points conversion is a documented heuristic,
NOT an empirically calibrated prediction or betting price.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from .core import TEAMS, canonical, fit_ratings, predict

REQUIRED = {'game_id','posteam','defteam','play_type','epa','success','yards_gained'}

def aggregate_pbp(pbp):
    """Create one row per game, offense and pass/run category; never use final score in features."""
    missing=REQUIRED-set(pbp.columns)
    if missing: raise ValueError(f'Play-by-play source missing fields: {sorted(missing)}')
    v=pbp[list(REQUIRED)].copy()
    v['posteam']=v['posteam'].map(canonical)
    v['defteam']=v['defteam'].map(canonical)
    v['epa']=pd.to_numeric(v['epa'],errors='coerce')
    v['success']=pd.to_numeric(v['success'],errors='coerce')
    v['yards_gained']=pd.to_numeric(v['yards_gained'],errors='coerce')
    v=v.loc[v.posteam.isin(TEAMS)&v.defteam.isin(TEAMS)&v.play_type.isin(['pass','run'])&v.epa.notna()]
    # nflverse calls rushes 'run'. Do not double count penalties / nonplays.
    v['unit']=v.play_type.map({'pass':'pass','run':'rush'})
    result=v.groupby(['game_id','posteam','defteam','unit'],as_index=False).agg(
        plays=('epa','size'),epa_per_play=('epa','mean'),success_rate=('success','mean'),
        yards_per_play=('yards_gained','mean'))
    return result.sort_values(['game_id','posteam','unit']).reset_index(drop=True)

def load_or_refresh_features(path,games,refresh=False):
    """Fetch necessary nflverse seasons if needed and persist compact aggregate cache."""
    path.parent.mkdir(parents=True,exist_ok=True)
    known=pd.read_csv(path) if path.exists() and not refresh else pd.DataFrame()
    if len(known) and {'game_id','posteam','defteam','unit','plays','epa_per_play'}.issubset(known.columns):
        seen=set(known.game_id.astype(str))
        if set(games.game_id.astype(str)).issubset(seen):return known
    import nflreadpy as nfl
    seasons=sorted(int(s) for s in games.season.unique())
    chunks=[]
    for season in seasons:
        wanted=set(games.loc[games.season==season,'game_id'].astype(str))
        print(f'  Fetching {season} play-by-play (nflverse)...',flush=True)
        raw=nfl.load_pbp(seasons=season)
        # Filter BEFORE pandas conversion to avoid gigabyte-scale allocations.
        raw=raw.filter(__import__('polars').col('game_id').cast(__import__('polars').String).is_in(list(wanted)))
        if raw.height:
            keep=list(REQUIRED)
            chunks.append(aggregate_pbp(raw.select(keep).to_pandas()))
    if not chunks:raise ValueError('No PBP rows matched historical completed-game IDs')
    output=pd.concat(chunks,ignore_index=True)
    output.to_csv(path,index=False)
    return output

def fit_epa(games,features,half_life_days=280,alpha=20):
    """Opponent-adjusted per-play efficiency split by pass/rush.

    Each unit regresses team offense + opponent defense, controlling for
    opposition and weighting both recent games and observed play counts.
    """
    joined=features.merge(games[['game_id','date']],on='game_id',how='inner')
    if len(joined)==0:raise ValueError('No play-by-play coverage for training games')
    ids={t:i for i,t in enumerate(TEAMS)}
    pieces={}; nteams=len(TEAMS)
    for unit in ('pass','rush'):
        sub=joined.loc[(joined.unit==unit)&(joined.plays>=3)].copy()
        if len(sub)<100:raise ValueError(f'Insufficient {unit} EPA observations: {len(sub)}')
        X=np.zeros((len(sub),2*nteams)); y=sub.epa_per_play.to_numpy(float)
        for k,r in enumerate(sub.itertuples()):
            X[k,ids[r.posteam]]=1
            X[k,nteams+ids[r.defteam]]=1
        d=(games.date.max()-sub.date).dt.days.to_numpy()
        weights=sub.plays.to_numpy(float)*np.power(.5,d/half_life_days)
        model=Ridge(alpha=alpha,fit_intercept=True).fit(X,y,sample_weight=weights)
        o=model.coef_[:nteams]; allowed=model.coef_[nteams:]
        pieces[unit]={'off':o-o.mean(),'allowed':allowed-allowed.mean(),
            'samples':len(sub),'plays':int(sub.plays.sum())}
    return pieces

def enhanced_ratings(games,features,blend=.32,pass_plays=37,rush_plays=25):
    """Keep underlying scoring model; add shrinkage-based EPA challenger.

    EPA contributions are centered across teams, so full-league scoring level
    and home field remain governed by the score model.
    """
    base,meta=fit_ratings(games)
    layers=fit_epa(games,features,meta['half_life_days'])
    rows=base.copy().set_index('team')
    for u in ('pass','rush'):
        pieces=layers[u]
        rows[f'{u}_off_epa']=pd.Series(pieces['off'],index=TEAMS)
        rows[f'{u}_def_epa_allowed']=pd.Series(pieces['allowed'],index=TEAMS)
    o=(rows.pass_off_epa*pass_plays+rows.rush_off_epa*rush_plays)
    d=(rows.pass_def_epa_allowed*pass_plays+rows.rush_def_epa_allowed*rush_plays)
    rows['score_only_overall']=rows.overall
    rows['epa_component_overall']=o-d
    rows['offense']= (1-blend)*rows.offense+blend*o
    rows['defense']= (1-blend)*rows.defense+blend*d
    rows['overall']=rows.offense-rows.defense
    rows['offense_points']=meta['baseline_points_per_team']+rows.offense
    rows['defense_allowed']=meta['baseline_points_per_team']+rows.defense
    for col,ascending,out in [('overall',False,'overall_rank'),('offense',False,'offense_rank'),('defense',True,'defense_rank')]:
        rows[out]=rows[col].rank(ascending=ascending,method='min').astype(int)
    rows=rows.reset_index().sort_values('overall',ascending=False).reset_index(drop=True)
    meta={**meta,'advanced':True,'epa_blend':blend,'epa_to_points_play_counts':{'pass':pass_plays,'rush':rush_plays},
        'feature_game_ids':int(features[features.game_id.isin(games.game_id)].game_id.nunique()),
        'epa_samples':{u:layers[u]['samples'] for u in layers},
        'epa_note':'Heuristic, uncalibrated pass/rush EPA adjustment; test against score-only baseline'}
    return rows,meta

def compare_backtest(games,features,holdout=128,minimum_train=800,blend=.32):
    """Walk-forward score-only vs EPA forecasts; same-day future games excluded."""
    rows=[]
    for date,block in games.tail(holdout).groupby('date',sort=True):
        prior=games[games.date<date]
        if len(prior)<minimum_train:continue
        old,om=fit_ratings(prior)
        newer,nm=enhanced_ratings(prior,features,blend=blend)
        for g in block.itertuples():
            a=predict(g.home_team,g.away_team,old,om)
            b=predict(g.home_team,g.away_team,newer,nm)
            rows.append({'date':str(date.date()),'game_id':g.game_id,'actual_margin':g.home_score-g.away_score,
                'actual_total':g.home_score+g.away_score,'score_margin':a['home_margin'],'epa_margin':b['home_margin'],
                'score_total':a['total'],'epa_total':b['total']})
    frame=pd.DataFrame(rows)
    if frame.empty:return {'games_tested':0},frame
    metrics={'games_tested':len(frame)}
    for name in ['score','epa']:
        metrics[name+'_margin_mae']=round(float((frame[name+'_margin']-frame.actual_margin).abs().mean()),3)
        metrics[name+'_total_mae']=round(float((frame[name+'_total']-frame.actual_total).abs().mean()),3)
    metrics['epa_margin_mae_improvement']=round(metrics['score_margin_mae']-metrics['epa_margin_mae'],3)
    metrics['epa_total_mae_improvement']=round(metrics['score_total_mae']-metrics['epa_total_mae'],3)
    return metrics,frame
