import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "stocksense_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"
IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"

FEATURE_COLUMNS = [
    "company_pe",
    "industry_pe",
    "pb_ratio",
    "roe",
    "roce",
    "debt_to_equity",
    "interest_coverage",
    "sales_growth_5y",
    "profit_growth_5y",
    "operating_profit_margin",
    "free_cash_flow",
    "promoter_holding",
    "promoter_pledge",
    "current_ratio",
    "eps_growth",
]


def load_training_data(path, target):
    data = pd.read_csv(path)
    missing = [column for column in FEATURE_COLUMNS + [target] if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    frame = data[FEATURE_COLUMNS + [target]].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        raise ValueError("No valid rows remain after cleaning missing and infinite values.")

    x = frame[FEATURE_COLUMNS].astype(float)
    y = frame[target].astype(float).clip(0, 100)
    return x, y


def build_search(random_state):
    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="mae",
        random_state=random_state,
        n_jobs=-1,
    )
    param_distributions = {
        "n_estimators": [300, 500, 750, 1000],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.015, 0.025, 0.04, 0.06, 0.08],
        "subsample": [0.70, 0.80, 0.90, 1.0],
        "colsample_bytree": [0.70, 0.80, 0.90, 1.0],
        "min_child_weight": [1, 3, 5, 8],
        "reg_alpha": [0, 0.01, 0.05, 0.1, 0.5],
        "reg_lambda": [0.5, 1, 1.5, 2, 4],
    }
    return RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=35,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )


def recommendation_bucket(score):
    if score >= 80:
        return "buy"
    if score >= 60:
        return "watchlist"
    return "avoid"


def bucket_accuracy(y_true, y_pred):
    true_bucket = [recommendation_bucket(value) for value in y_true]
    pred_bucket = [recommendation_bucket(value) for value in y_pred]
    return float(np.mean(np.array(true_bucket) == np.array(pred_bucket)))


def train(args):
    x, y = load_training_data(args.data, args.target)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    if args.search:
        search = build_search(args.random_state)
        search.fit(x_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=650,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_alpha=0.05,
            reg_lambda=1.5,
            tree_method="hist",
            eval_metric="mae",
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        best_params = model.get_params()

    predictions = np.clip(model.predict(x_test), 0, 100)
    metadata = {
        "model": type(model).__name__,
        "target": args.target,
        "rows_after_cleaning": int(len(x)),
        "features": FEATURE_COLUMNS,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "metrics": {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
            "bucket_accuracy": bucket_accuracy(y_test, predictions),
        },
        "params": best_params,
        "note": (
            "Bucket accuracy measures whether predicted scores fall into the same "
            "buy/watchlist/avoid range as the labels. For real-world claims, validate "
            "against future returns, not only synthetic labels."
        ),
    }

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURES_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    importance.to_csv(IMPORTANCE_PATH, index=False)

    print(json.dumps(metadata["metrics"], indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Saved feature importance to {IMPORTANCE_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the StockSense AI XGBoost model.")
    parser.add_argument("--data", required=True, help="Path to a CSV training dataset.")
    parser.add_argument("--target", default="score", help="Target score column, default: score.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--search", action="store_true", help="Run RandomizedSearchCV for stronger tuning.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
