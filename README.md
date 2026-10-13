# Grizz NFL V0.4 — Cloud-ready

Clean Streamlit app plus scheduled GitHub Actions update. **No published live site yet**, and market integration and moneyline pricing are not connected.

## Production model policy
- **Margins/spreads:** V0.1 score-only baseline, best on the supplied 137-game holdout (MAE 10.399).
- **Totals:** V0.2 32%-EPA blend, best on that holdout (MAE 10.826).
- **Power ratings:** V0.2 32%-EPA blend, descriptive only, not equivalent to chosen margin model.
- **Moneylines:** No probabilities displayed; Brier/log loss are not strong enough for market pricing.
- No lookahead: `load_games` trains on dates strictly before today's UTC date. Today's completed games appear the next UTC day. 

## Launch the cloud site
1. Create a new **GitHub repository** and upload **all files including the `.github` and `.streamlit` folders** at the repo root. GitHub Actions scheduled runs only occur on the default branch. If your repository has Actions disabled, enable them in GitHub Settings.
2. Go to Actions > **Update Grizz Ratings** > **Run workflow**. This downloads game results and relevant play-by-play, computes the rankings, then commits `output/current.json`. If it fails, inspect the job log; stale results are never silently published as new.
3. At https://share.streamlit.io create a Streamlit app linked to the repository, main file **`streamlit_app.py`**. Select Python 3.11 if needed. Your hosted URL will appear after deployment.
4. The workflow also refreshes daily at 11:17 UTC (scheduled jobs may start late or skip under platform load); use the manual workflow trigger anytime.

## Optional local preview (setup only once)
```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe update.py
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

On subsequent runs, only `python update.py` is needed if you want to refresh locally. Hosting removes the need entirely.

## Caveats
- To minimize costs, the workflow publishes a committed JSON snapshot; `data/games.csv` and large play-by-play caches are ignored, with a GitHub Actions cache to help future runs.
- The initial download may take time. GitHub Actions has runtime/usage quotas and cannot guarantee an exact schedule.
- One real-data run of this NEW V0.4 package has not been completed here. The site shows an explicit pending state until GitHub produces `output/current.json`.
- Today's pregame changes, starting QBs, injury feeds, market snapshots, prediction version persistence, and season-spanning walk-forward tuning are future additions.
- No stored odds, fair betting probabilities, automated wagering, or proof of positive EV.
