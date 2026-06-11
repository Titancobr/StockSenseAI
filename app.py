from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, abort, render_template, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "stocksense_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.pkl"

FIELD_DEFINITIONS = [
    ("Stock P/E", "company_pe", "e.g. 22.5"),
    ("Industry PE", "industry_pe", "e.g. 28.0"),
    ("Price to book value", "pb_ratio", "e.g. 3.2"),
    ("ROE", "roe", "e.g. 18.5"),
    ("ROCE", "roce", "e.g. 21.4"),
    ("Debt to equity", "debt_to_equity", "e.g. 0.35"),
    ("Int Coverage", "interest_coverage", "e.g. 8.2"),
    ("Sales growth 5Years", "sales_growth_5y", "e.g. 14.5"),
    ("Profit Var 5Yrs", "profit_growth_5y", "e.g. 16.8"),
    ("OPM", "operating_profit_margin", "e.g. 19.2"),
    ("Free Cash Flow", "free_cash_flow", "e.g. 850"),
    ("Promoter holding", "promoter_holding", "e.g. 62.0"),
    ("Pledged percentage", "promoter_pledge", "e.g. 0"),
    ("Current ratio", "current_ratio", "e.g. 1.8"),
    ("EPS growth 3Years", "eps_growth", "e.g. 17.5"),
]

app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
    feature_columns = list(joblib.load(FEATURES_PATH))
except Exception as exc:
    raise RuntimeError(f"Unable to load the uploaded StockSense AI model: {exc}") from exc

expected_features = [field[1] for field in FIELD_DEFINITIONS]
if feature_columns != expected_features:
    raise RuntimeError(
        "feature_columns.pkl does not match the required StockSense input mapping."
    )


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/guide")
def guide():
    return render_template("guide.html")


@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    if request.method == "GET":
        return render_template("analysis.html", fields=FIELD_DEFINITIONS, values={})

    values = {}
    errors = {}
    for label, feature, _ in FIELD_DEFINITIONS:
        raw_value = request.form.get(feature, "").strip()
        values[feature] = raw_value
        try:
            number = float(raw_value)
            if not pd.notna(number):
                raise ValueError
            values[feature] = number
        except (TypeError, ValueError):
            errors[feature] = f"Enter a valid numeric value for {label}."

    if errors:
        return (
            render_template(
                "analysis.html",
                fields=FIELD_DEFINITIONS,
                values=values,
                errors=errors,
            ),
            400,
        )

    input_frame = pd.DataFrame(
        [[values[column] for column in feature_columns]],
        columns=feature_columns,
    )

    try:
        raw_score = float(model.predict(input_frame)[0])
    except Exception as exc:
        app.logger.exception("Model prediction failed")
        return render_template("error.html", message=str(exc)), 500

    score = max(0, min(100, round(raw_score)))
    if score >= 80:
        recommendation = "BUY"
        tone = "buy"
        explanation = (
            "The stock appears fundamentally strong according to the AI model and "
            "can be considered a strong long-term candidate."
        )
    elif score >= 60:
        recommendation = "WATCHLIST"
        tone = "watchlist"
        explanation = (
            "The stock has moderate fundamentals. Keep an eye on it and analyze it "
            "again in the future."
        )
    else:
        recommendation = "AVOID"
        tone = "avoid"
        explanation = (
            "The stock appears fundamentally weak according to the AI model. It is "
            "better to avoid investing unless the fundamentals improve significantly."
        )

    return render_template(
        "result.html",
        score=score,
        recommendation=recommendation,
        tone=tone,
        explanation=explanation,
    )


@app.get("/media/<filename>")
def media(filename):
    if filename not in {"install_screener.mp4", "how_to_use.mp4"}:
        abort(404)
    return send_from_directory(BASE_DIR, filename, conditional=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": type(model).__name__, "features": len(feature_columns)}


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", message="The page you requested could not be found."), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
