import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title='GRIZZ | NFL Models',page_icon='🏈',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
#MainMenu,footer {visibility:hidden}
.block-container{max-width:1280px;padding-top:2.4rem;padding-bottom:4rem}
[data-testid="stAppViewContainer"]{background:#0a101b;color:#e9edf5}
h1,h2,h3{letter-spacing:-.04em}
.grizz-logo{font-size:2.2rem;font-weight:900;letter-spacing:-.06em;color:#f5f8ff}
.grizz-tag{color:#8899b3;font-size:.9rem;margin-top:-.4rem}
.panel{border:1px solid #273348;background:#111b2c;border-radius:14px;padding:18px;margin:8px 0}
.small{color:#a1adc2;font-size:.82rem}
.big{font-size:1.6rem;font-weight:800;margin-top:4px}
[data-testid="stMetric"]{background:#111b2c;border:1px solid #253449;border-radius:14px;padding:14px}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
.stTabs [data-baseweb="tab-list"]{gap:22px}
.stTabs [data-baseweb="tab"]{font-weight:700}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="grizz-logo">GRIZZ<span style="color:#59d6c0">.</span></div><div class="grizz-tag">NFL predictive analytics / Power ratings</div>',unsafe_allow_html=True)
p=Path(__file__).resolve().parent/'output/current.json'
if not p.exists():
    st.warning('Ratings pending first verified cloud refresh. No demo picks or fabricated rankings are being displayed.')
    st.markdown('Run `python update.py` or execute the **Update Grizz Ratings** GitHub workflow. Then redeploy or refresh this page.')
    st.stop()
try:data=json.loads(p.read_text(encoding='utf8'))
except Exception as exc:st.error(f'Unable to read latest data: {exc}');st.stop()
st.caption(f'Data through {data["last_game"]}  ·  {data["training_games"]:,} completed NFL games  ·  Refreshed {data["generated_utc"][:16]} UTC')
tab1,tab2,tab3=st.tabs(['MODEL PICKS','GRIZZ RATINGS','MODEL PERFORMANCE'])
with tab1:
    st.subheader('Game projections')
    st.caption('Spreads use the validated score-only benchmark; totals use the EPA challenger. No odds, probabilities or +EV claims.')
    a,b,c=st.columns(3);a.metric('Projected games',len(data['picks']));b.metric('Margin model','V0.1');c.metric('Total model','V0.2')
    df=pd.DataFrame(data['picks'])
    if df.empty:st.info('No upcoming games present in the source schedule.')
    else:
        date_range=st.selectbox('Games', ['All upcoming games']+sorted(df['date'].unique().tolist()))
        if date_range!='All upcoming games':df=df[df['date']==date_range]
        df=df.rename(columns={'date':'Date','matchup':'Matchup','away_points':'Away pts','home_points':'Home pts','home_margin':'Home margin','total':'Total'})
        st.dataframe(df[['Date','Matchup','Away pts','Home pts','Home margin','Total']],hide_index=True,use_container_width=True)
        st.caption('Home margin = home projected points minus away projected points. Positive means home favored.')
with tab2:
    st.subheader('Grizz power ratings')
    st.caption('Opponent-adjusted. Higher offense and overall are better; lower defense points allowed is better.')
    r=pd.DataFrame(data['ratings'])
    if r.empty:st.warning('No ratings found')
    else:
        r['Team']=r['team_name']+' · '+r['team'];r=r.rename(columns={'overall_rank':'Rank','overall':'Overall','offense_points':'Offense','offense_rank':'Off. rank','defense_allowed':'Defense allowed','defense_rank':'Def. rank'})
        st.dataframe(r[['Rank','Team','Overall','Offense','Off. rank','Defense allowed','Def. rank']],hide_index=True,use_container_width=True,height=820,
            column_config={'Overall':st.column_config.NumberColumn(format='%+.2f'),'Offense':st.column_config.NumberColumn(format='%.2f'),'Defense allowed':st.column_config.NumberColumn(format='%.2f')})
        st.caption('Ratings are descriptive EPA-blended estimates; spread projections intentionally use the V0.1 score-only benchmark.')
with tab3:
    st.subheader('Performance tracking')
    st.caption('Frozen 137-game test reference supplied from V0.3 comparison; not rerun during daily refresh.')
    m=data['test_reference'];a,b,c=st.columns(3)
    a.metric('Best spread MAE',f'{m["v01_margin_mae"]:.3f} pts');b.metric('Best total MAE',f'{m["v02_total_mae"]:.3f} pts');c.metric('Best winner Brier',f'{m["v01_winner_brier"]:.4f}')
    st.info('These are historical benchmarks, not evidence of profitable picks. Moneyline and spread-cover probabilities remain withheld until proper calibration and market comparison.')
    st.markdown('**Current production policy**')
    st.write('Margin: V0.1 score-only · Total: V0.2 fixed EPA 32% · Ratings: V0.2 descriptive · Market feeds: not connected.')
