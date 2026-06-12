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

FEATURE_LABELS = {feature: label for label, feature, _ in FIELD_DEFINITIONS}

PARAMETER_GROUPS = [
    (
        "Valuation discipline",
        "Checks whether the stock is priced reasonably compared with industry and book value.",
        ["company_pe", "industry_pe", "pb_ratio"],
    ),
    (
        "Profitability quality",
        "Rewards companies that generate strong returns on equity and capital employed.",
        ["roe", "roce", "operating_profit_margin"],
    ),
    (
        "Balance sheet safety",
        "Penalizes high debt and weak liquidity while rewarding interest coverage strength.",
        ["debt_to_equity", "interest_coverage", "current_ratio"],
    ),
    (
        "Growth consistency",
        "Looks for durable sales, profit, and EPS growth over multi-year periods.",
        ["sales_growth_5y", "profit_growth_5y", "eps_growth"],
    ),
    (
        "Cash and ownership quality",
        "Combines free cash flow, promoter ownership, and pledge risk.",
        ["free_cash_flow", "promoter_holding", "promoter_pledge"],
    ),
]

LITERATURE_REFERENCES = [
    {
        "title": "Machine Learning for Stock Prediction Based on Fundamental Analysis",
        "authors": "Huang, Capretz and Ho",
        "year": "2021",
        "takeaway": "Shows that ML models trained on quarterly financial fundamentals can support fundamental analysts in stock investment decisions.",
    },
    {
        "title": "A Practical Machine Learning Approach for Dynamic Stock Recommendation",
        "authors": "Yang, Liu and Wu",
        "year": "2025",
        "takeaway": "Frames stock screening as a ranking and recommendation task using financial indicators and rolling model selection.",
    },
    {
        "title": "Stock Market Prediction Using Machine Learning and Deep Learning Techniques: A Review",
        "authors": "Saberironaghi, Ren and Saberironaghi",
        "year": "2025",
        "takeaway": "Highlights common finance-ML challenges such as data quality, interpretability, model evaluation, and dataset choice.",
    },
]

MODEL_LOAD_ERROR = None

app = Flask(__name__)

try:
    feature_columns = list(joblib.load(FEATURES_PATH))
except Exception as exc:
    raise RuntimeError(f"Unable to load the StockSense feature mapping: {exc}") from exc

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    model = None
    MODEL_LOAD_ERROR = str(exc)

expected_features = [field[1] for field in FIELD_DEFINITIONS]
if feature_columns != expected_features:
    raise RuntimeError(
        "feature_columns.pkl does not match the required StockSense input mapping."
    )


def _bounded_score(value, low, high, invert=False):
    if high == low:
        return 0
    score = (value - low) / (high - low) * 100
    if invert:
        score = 100 - score
    return max(0, min(100, score))


def _fallback_stock_score(values):
    valuation_gap = values["company_pe"] / values["industry_pe"] if values["industry_pe"] > 0 else 2
    category_scores = {
        "Valuation discipline": round(
            0.60 * _bounded_score(valuation_gap, 0.45, 1.30, invert=True)
            + 0.40 * _bounded_score(values["pb_ratio"], 1.0, 7.0, invert=True)
        ),
        "Profitability quality": round(
            0.35 * _bounded_score(values["roe"], 8, 28)
            + 0.40 * _bounded_score(values["roce"], 10, 35)
            + 0.25 * _bounded_score(values["operating_profit_margin"], 8, 30)
        ),
        "Balance sheet safety": round(
            0.40 * _bounded_score(values["debt_to_equity"], 0, 1.5, invert=True)
            + 0.35 * _bounded_score(values["interest_coverage"], 1.5, 15)
            + 0.25 * _bounded_score(values["current_ratio"], 0.8, 2.5)
        ),
        "Growth consistency": round(
            0.34 * _bounded_score(values["sales_growth_5y"], 0, 25)
            + 0.33 * _bounded_score(values["profit_growth_5y"], 0, 30)
            + 0.33 * _bounded_score(values["eps_growth"], 0, 28)
        ),
        "Cash and ownership quality": round(
            0.35 * _bounded_score(values["free_cash_flow"], -500, 3000)
            + 0.40 * _bounded_score(values["promoter_holding"], 25, 75)
            + 0.25 * _bounded_score(values["promoter_pledge"], 0, 25, invert=True)
        ),
    }
    weights = {
        "Valuation discipline": 0.18,
        "Profitability quality": 0.25,
        "Balance sheet safety": 0.22,
        "Growth consistency": 0.22,
        "Cash and ownership quality": 0.13,
    }
    score = sum(category_scores[name] * weight for name, weight in weights.items())
    return score, category_scores


def _category_scores_from_values(values):
    _score, category_scores = _fallback_stock_score(values)
    return category_scores


def _build_stock_diagnostics(values, category_scores):
    strengths = []
    watchouts = []

    if values["roce"] >= 20:
        strengths.append("ROCE indicates efficient use of capital.")
    if values["roe"] >= 15:
        strengths.append("ROE suggests healthy shareholder returns.")
    if values["debt_to_equity"] <= 0.5:
        strengths.append("Debt-to-equity is conservative for a long-term holding.")
    if values["interest_coverage"] >= 8:
        strengths.append("Interest coverage gives the business a stronger safety buffer.")
    if values["promoter_pledge"] <= 1:
        strengths.append("Promoter pledge risk is low.")
    if values["profit_growth_5y"] >= 12 and values["sales_growth_5y"] >= 10:
        strengths.append("Sales and profit growth appear consistent.")

    valuation_gap = values["company_pe"] / values["industry_pe"] if values["industry_pe"] > 0 else None
    if valuation_gap is not None and valuation_gap > 1.15:
        watchouts.append("Company P/E is meaningfully above industry P/E, so valuation needs extra caution.")
    if values["debt_to_equity"] > 1:
        watchouts.append("Debt-to-equity is elevated and may increase downside risk.")
    if values["interest_coverage"] < 3:
        watchouts.append("Interest coverage is weak, which can become dangerous during rate hikes.")
    if values["current_ratio"] < 1:
        watchouts.append("Current ratio below 1 may indicate short-term liquidity pressure.")
    if values["promoter_pledge"] > 5:
        watchouts.append("Promoter pledge is visible and should be investigated before investing.")
    if values["free_cash_flow"] < 0:
        watchouts.append("Negative free cash flow weakens the long-term quality signal.")

    strengths = strengths[:4] or ["The model did not find a standout fundamental strength; compare with peers before acting."]
    watchouts = watchouts[:4] or ["No major red flag was triggered by the entered values, but news and sector context still matter."]

    sorted_categories = sorted(category_scores.items(), key=lambda item: item[1])
    weakest_category = sorted_categories[0][0]
    strongest_category = sorted_categories[-1][0]

    return {
        "strengths": strengths,
        "watchouts": watchouts,
        "strongest_category": strongest_category,
        "weakest_category": weakest_category,
    }


@app.get("/")
def home():
    return render_template(
        "index.html",
        parameter_groups=PARAMETER_GROUPS,
        literature=LITERATURE_REFERENCES,
    )


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

    category_scores = _category_scores_from_values(values)

    if model is None:
        raw_score, category_scores = _fallback_stock_score(values)
        engine_label = "Transparent fallback scorer"
        engine_note = (
            "The uploaded XGBoost model could not be loaded in this local environment, "
            "so this run used the documented rule-based fallback. Install the dependencies "
            "from requirements.txt to use the trained model."
        )
    else:
        try:
            raw_score = float(model.predict(input_frame)[0])
            engine_label = "Uploaded XGBoost model"
            engine_note = (
                "The score was generated by the trained StockSense AI model. The category "
                "breakdown below is a transparent explanation layer for review and demos."
            )
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
        engine_label=engine_label,
        engine_note=engine_note,
        category_scores=category_scores,
        diagnostics=_build_stock_diagnostics(values, category_scores),
        literature=LITERATURE_REFERENCES,
    )


@app.get("/media/<filename>")
def media(filename):
    if filename not in {"install_screener.mp4", "how_to_use.mp4"}:
        abort(404)
    return send_from_directory(BASE_DIR, filename, conditional=True)


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "model": type(model).__name__ if model is not None else "unavailable",
        "model_load_error": MODEL_LOAD_ERROR,
        "features": len(feature_columns),
    }


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", message="The page you requested could not be found."), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
