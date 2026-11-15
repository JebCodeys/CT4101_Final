# CT4101: Flow-level intrusion detection on CIC-IDS2017

Multiclass classification of network flows into 7 families (Benign, DoS, DDoS, PortScan, BruteForce, WebAttack, Bot) from the 78 CICFlowMeter features in the MachineLearningCVE release.

## Setup
```
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Put the 8 MachineLearningCVE CSVs from https://www.unb.ca/cic/datasets/ids-2017.html in `data/raw/`.

## Milestone 2: data and baseline checkpoint
`notebooks/milestone2_data_and_baseline.ipynb` runs end to end, in ~30 min on 2 cores / 7 GB RAM. Peak memory is ~4.5 GB.

## Milestone 3: model comparison checkpoint
`notebooks/milestone3_model_comparison.ipynb` compares the milestone-2 baselines with a random forest and a regularised histogram gradient-boosting model, each trained on the natural traffic mix and on a **worst-case background** (every session's benign traffic re-weighted up to the volume of Monday, the heaviest attack-free day, so no attack family is learned against a quiet afternoon). It then re-scores the out-of-fold predictions under a back-to-back multi-attack day, measures the false-alarm rate on benign traffic from a held-out day, and ends with an error analysis. Runtime ~2.2 h on 10 cores. The first run caches the cleaned splits to `data/*.parquet` (git-ignored).

## Layout
- `src/data.py`: loading, label normalisation, audit, redundant-column detection, two-stage dedup, conflict detection
- `src/features.py`: preprocessing `ColumnTransformer` (negative→NaN, median imputation with indicators, signed log, scaling; one-hot port)
- `src/evaluate.py`: stratified k-fold CV loop (optional training-fold sample weights) with per-fold metrics and out-of-fold predictions; scenario-weighted class report; held-out-day benign fit
- `src/background.py`: per-session background table, worst-case-background sample weights, scenario importance weights
- `reports/`: audit JSON, CV results JSON (`baselines_cv.json`, `m3_model_comparison.json`), figures

The 20% test split (stratified, seed 4101) is never scored before final model selection.
