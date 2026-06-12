# StockSense AI

AI-assisted fundamental stock screening for Indian long-term investors.

StockSense AI accepts 15 company fundamentals, validates every input, and
returns a 0-100 fundamental strength score with a simple recommendation:

- `80-100`: Buy candidate
- `60-79`: Watchlist
- `<60`: Avoid

The project is positioned as **ML-assisted fundamental screening**, not
short-term price prediction and not investment advice.

## What is included

- Flask backend with the uploaded `stocksense_model.pkl` scorer.
- Beginner guide videos for collecting required values.
- 15-parameter analysis form.
- Result page with score gauge, recommendation, explanation, strengths,
  watchouts, and category-level diagnostics.
- Graceful fallback scorer for local demo/test environments where `xgboost` is
  not installed.
- XGBoost retraining script with optional hyperparameter search.
- Current model card and feature-importance export.
- LaTeX hackathon deck at `docs/StockSenseAI_Hackathon_Deck.tex`.

## Methodology

1. Selected 15 fundamental indicators used in long-term stock analysis.
2. Generated a synthetic dataset from expert-defined investment rules.
3. Cleaned the data and trained an XGBoost model.
4. Integrated the trained model into a Flask web app.
5. Added beginner onboarding, result thresholds, risk alerts, and explanation
   panels.

Use this wording when presenting the result:

> The model achieved 94% classification accuracy on a synthetic
> fundamental-analysis dataset generated from expert-defined investment rules.

Do not claim 94% real-world stock-picking accuracy without historical return
validation.

## Literature base

- Huang, Capretz and Ho, "Machine Learning for Stock Prediction Based on
  Fundamental Analysis", IEEE SSCI, 2021.
- Yang, Liu and Wu, "A Practical Machine Learning Approach for Dynamic Stock
  Recommendation", arXiv:2511.12129.
- Saberironaghi, Ren and Saberironaghi, "Stock Market Prediction Using Machine
  Learning and Deep Learning Techniques: A Review", AppliedMath, 2025.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Test

```bash
python -m unittest
```

## Retrain the XGBoost model

Prepare a CSV with these 15 feature columns and a target score column named
`score`:

```text
company_pe, industry_pe, pb_ratio, roe, roce, debt_to_equity,
interest_coverage, sales_growth_5y, profit_growth_5y,
operating_profit_margin, free_cash_flow, promoter_holding,
promoter_pledge, current_ratio, eps_growth, score
```

Fast training:

```bash
python train_model.py --data path/to/training_data.csv
```

Stronger tuned training for the final hackathon model:

```bash
python train_model.py --data path/to/training_data.csv --search
```

The script saves:

- `stocksense_model.pkl`
- `feature_columns.pkl`
- `model_metadata.json`
- `feature_importance.csv`

For the strongest project defense, retrain on historical real-company
fundamentals and validate the score buckets against future 1-year, 3-year, and
5-year returns.

## Included dataset

This repo includes a generated training dataset at:

```text
data/training_data.csv
```

It was created with:

```bash
python generate_synthetic_dataset.py
```

The generated dataset contains 62,500 synthetic rows that imitate multiple
fundamental profiles: quality compounders, average companies, expensive quality
stocks, cyclicals, distressed companies, debt traps, pledge-risk companies,
turnaround cases, and negative cash-flow growth cases. It is useful for
hackathon demos and repeatable model training, but it should be described
honestly as synthetic expert-rule data. Replace or augment it with real
historical fundamentals before claiming real-world investment performance.
