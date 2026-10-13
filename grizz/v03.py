"""V0.3: temporal validation, separate margin/total blends, held-out evaluation.

Ratings can still be reported from the EPA model, but prediction weights are selected
without looking at the final test window. Probabilities use normal residual approximation
fitted on validation only. These are experimental, not market-ready.
"""
import math
import numpy as np
import pandas as pd
from .core import fit_ratings, predict
from .advanced import enhanced_ratings

WEIGHTS=[0, .1, .2, .3, .4, .5, .65, .8, 1.0]

def replay_candidates(games, features, n_validation=128, n_test=128, minimum_train=800):
    if len(games)<n_validation+n_test+minimum_train:
        raise ValueError('Not enough games for requested validation and test sets')
    # Select complete calendar dates BEFORE forecasting. Taking only the last
    # n_validation+n_test rows can truncate the earlier validation window when
    # the test boundary expands to include an entire NFL game day.
    ordered=games.sort_values(['date','game_id']).reset_index(drop=True)
    test_start=ordered.iloc[-n_test]['date']
    pre=ordered.loc[ordered.date<test_start]
    if len(pre)<n_validation:
        raise ValueError('Not enough games before the test window for validation')
    validation_start=pre.iloc[-n_validation]['date']
    evalgames=ordered.loc[ordered.date>=validation_start]
    expected_test=int((evalgames.date>=test_start).sum())
    expected_val=int((evalgames.date<test_start).sum())
    if expected_test>n_test:
        print(f'Whole-date test boundary expanded test to {expected_test} games')
    if expected_val>n_validation:
        print(f'Whole-date validation boundary expanded validation to {expected_val} games')
    if len(ordered.loc[ordered.date<validation_start])<minimum_train:
        raise ValueError('Not enough earlier games for leakage-safe training before validation')
    frames=[]
    for date, group in evalgames.groupby('date',sort=True):
        prior=ordered.loc[ordered.date<date]
        if len(prior)<minimum_train:
            raise ValueError(f'Insufficient training history before {date}')
        score,sm=fit_ratings(prior)
        epa,em=enhanced_ratings(prior,features,blend=1.0)
        for g in group.itertuples():
            s=predict(g.home_team,g.away_team,score,sm)
            e=predict(g.home_team,g.away_team,epa,em)
            frames.append({'date':str(date.date()),'game_id':g.game_id,
              'home':g.home_team,'away':g.away_team,
              'actual_margin':float(g.home_score-g.away_score),
              'actual_total':float(g.home_score+g.away_score),
              'score_margin':s['home_margin'],'score_total':s['total'],
              'epa_margin':e['home_margin'],'epa_total':e['total']})
    frame=pd.DataFrame(frames)
    if len(frame)!=len(evalgames):
        raise ValueError('Historical replay did not cover every selected game')
    test=frame.loc[frame.date>=str(test_start.date())].copy()
    validation=frame.loc[frame.date<str(test_start.date())].copy()
    if validation.empty or test.empty:
        raise ValueError('Validation or test partition was unexpectedly empty')
    return validation.reset_index(drop=True),test.reset_index(drop=True)

def choose_weight(df, metric, weights=WEIGHTS):
    actual=df['actual_'+metric].to_numpy(float)
    s=df['score_'+metric].to_numpy(float)
    e=df['epa_'+metric].to_numpy(float)
    errors={float(w):float(np.mean(abs((1-w)*s+w*e-actual))) for w in weights}
    # Tie breaks toward original score model for conservative deployment.
    return min(errors,key=lambda w:(round(errors[w],8),w)), errors

def predict_blended(frame, margin_weight, total_weight):
    f=frame.copy()
    f['pred_margin']=(1-margin_weight)*f.score_margin+margin_weight*f.epa_margin
    f['pred_total']=(1-total_weight)*f.score_total+total_weight*f.epa_total
    return f

def fit_sigma(errors):
    errors=np.asarray(errors,float)
    # Root mean square error includes any systematic bias; do not fit on the test sample.
    return float(max(6,np.sqrt(np.mean(np.square(errors)))))

def normal_cdf(x):
    arr=np.asarray(x,dtype=float)
    return np.asarray([0.5*(1+math.erf(float(v)/math.sqrt(2))) for v in arr.flat]).reshape(arr.shape)

def win_probs(margins,sigma):
    # Continuity threshold for integer final NFL scoring margins.
    return normal_cdf((np.asarray(margins,float)-0.5)/sigma)

def metrics(df, sigma=None):
    out={}
    for m in ['margin','total']:
        err=(df['pred_'+m]-df['actual_'+m]).to_numpy(float)
        out[m+'_mae']=round(float(np.mean(abs(err))),3)
        out[m+'_rmse']=round(float(np.sqrt(np.mean(err**2))),3)
    if sigma is not None:
        observed=(df.actual_margin>0).astype(float).to_numpy()
        # Exclude ties from binary win accuracy and calibration.
        non_tie=(df.actual_margin!=0).to_numpy()
        p=win_probs(df.pred_margin.to_numpy(),sigma)
        out['winner_brier_excluding_ties']=round(float(np.mean((p[non_tie]-observed[non_tie])**2)),4) if non_tie.any() else None
        out['winner_log_loss_excluding_ties']=round(float(-np.mean(observed[non_tie]*np.log(np.clip(p[non_tie],1e-6,1-1e-6))+(1-observed[non_tie])*np.log(np.clip(1-p[non_tie],1e-6,1-1e-6)))),4) if non_tie.any() else None
    return out

def evaluate(games,features,n_validation=128,n_test=128):
    validation,test=replay_candidates(games,features,n_validation,n_test)
    wm,vm=choose_weight(validation,'margin')
    wt,vt=choose_weight(validation,'total')
    val=predict_blended(validation,wm,wt)
    sigma=fit_sigma(val.pred_margin-val.actual_margin)
    tested=predict_blended(test,wm,wt)
    baseline=predict_blended(test,0,0)
    old_32=predict_blended(test,.32,.32)
    result={'validation_games':len(val),'test_games':len(tested),
      'validation_first_date':val.date.min(),'validation_last_date':val.date.max(),
      'test_first_date':tested.date.min(),'test_last_date':tested.date.max(),
      'margin_epa_weight':wm,'total_epa_weight':wt,'validation_margin_mae_by_weight':vm,
      'validation_total_mae_by_weight':vt,'margin_residual_sigma':round(sigma,3),
      'validation_metrics':metrics(val,sigma),
      'test_v03':metrics(tested,sigma),'test_v01':metrics(baseline,sigma),'test_v02_fixed_32pct':metrics(old_32,sigma),
      'disclaimer':'Weights and sigma fit on validation only. Test is untouched during model selection. Binary winner probabilities are approximations; spreads, pushes and market pricing require further evaluation.'}
    return result, tested, val

def projected_win_probability(home_margin,sigma):
    return float(win_probs([home_margin],sigma)[0])

def cover_probability(home_margin,spread_home,sigma):
    # P(home margin + home spread > 0); discrete push mass not captured by normal approximation.
    return float(normal_cdf([(home_margin+spread_home-0.5)/sigma])[0])
