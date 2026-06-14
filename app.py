import json
import os
import sqlite3
from functools import wraps
from pathlib import Path
from urllib import request as urlrequest

import joblib
import pandas as pd
from flask import (
    Flask,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from authlib.integrations.flask_client import OAuth
except ImportError:  # pragma: no cover - lets the app explain missing optional deps.
    OAuth = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "stocksense_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.pkl"
DATABASE_PATH = Path(os.environ.get("STOCKSENSE_DATABASE", BASE_DIR / "stocksense.db"))
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-before-production")
LLM_MODEL = os.environ.get("STOCKSENSE_LLM_MODEL", "llama3.1")
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LLM_ENABLED = os.environ.get("STOCKSENSE_ENABLE_LLM", "").lower() in {"1", "true", "yes"}

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
app.config["SECRET_KEY"] = SECRET_KEY
oauth = OAuth(app) if OAuth is not None else None

if oauth and os.environ.get("GOOGLE_CLIENT_ID") and os.environ.get("GOOGLE_CLIENT_SECRET"):
    oauth.register(
        name="google",
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

if oauth and os.environ.get("APPLE_CLIENT_ID") and os.environ.get("APPLE_CLIENT_SECRET"):
    oauth.register(
        name="apple",
        client_id=os.environ["APPLE_CLIENT_ID"],
        client_secret=os.environ["APPLE_CLIENT_SECRET"],
        server_metadata_url="https://appleid.apple.com/.well-known/openid-configuration",
        client_kwargs={"scope": "name email"},
    )

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


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def init_db():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = get_db()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            name TEXT,
            password_hash TEXT,
            auth_provider TEXT NOT NULL DEFAULT 'password',
            provider_user_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            stock_name TEXT NOT NULL,
            input_values TEXT NOT NULL,
            score INTEGER NOT NULL,
            recommendation TEXT NOT NULL,
            explanation TEXT NOT NULL,
            category_scores TEXT NOT NULL,
            diagnostics TEXT NOT NULL,
            llm_source TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """
    )
    db.commit()


@app.before_request
def load_logged_in_user():
    init_db()
    user_id = session.get("user_id")
    g.user = None
    if user_id is not None:
        g.user = get_db().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


@app.teardown_appcontext
def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


@app.context_processor
def inject_current_user():
    return {"current_user": g.get("user")}


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            flash("Log in or create an account to save customer-specific analyses.")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped_view


def _find_user_by_email(email):
    return get_db().execute("SELECT * FROM users WHERE lower(email) = lower(?)", (email,)).fetchone()


def _create_or_update_oauth_user(provider, profile):
    email = (profile.get("email") or "").strip().lower()
    if not email:
        raise ValueError("The identity provider did not return an email address.")

    name = profile.get("name") or profile.get("given_name") or email.split("@")[0]
    provider_user_id = profile.get("sub") or profile.get("id")
    db = get_db()
    user = _find_user_by_email(email)
    if user is None:
        cursor = db.execute(
            """
            INSERT INTO users (email, name, auth_provider, provider_user_id)
            VALUES (?, ?, ?, ?)
            """,
            (email, name, provider, provider_user_id),
        )
        db.commit()
        return cursor.lastrowid

    db.execute(
        """
        UPDATE users
        SET name = COALESCE(?, name),
            auth_provider = ?,
            provider_user_id = COALESCE(?, provider_user_id)
        WHERE id = ?
        """,
        (name, provider, provider_user_id, user["id"]),
    )
    db.commit()
    return user["id"]


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


def _tone_for_recommendation(recommendation):
    normalized = recommendation.upper()
    if "BUY" in normalized:
        return "buy"
    if "AVOID" in normalized or "SELL" in normalized:
        return "avoid"
    return "watchlist"


def _fallback_recommendation(values, score, category_scores, diagnostics, stock_name):
    if score >= 80:
        recommendation = "BUY"
    elif score >= 60:
        recommendation = "WATCHLIST"
    else:
        recommendation = "AVOID"

    company = stock_name or "This stock"
    strongest = diagnostics["strongest_category"].lower()
    weakest = diagnostics["weakest_category"].lower()
    explanation = (
        f"{company} scores {score}/100. The strongest signal is {strongest}, while "
        f"{weakest} deserves the closest review before any decision."
    )
    description = (
        f"{company} shows {diagnostics['strengths'][0].lower()} The main caution is that "
        f"{diagnostics['watchouts'][0].lower()} Recheck quarterly results, sector news, and "
        "management commentary before treating this as an investment candidate."
    )
    return {
        "recommendation": recommendation,
        "tone": _tone_for_recommendation(recommendation),
        "explanation": explanation,
        "description": description,
        "source": "Rule-based fallback explanation",
    }


def _build_llm_prompt(values, score, category_scores, diagnostics, stock_name):
    company = stock_name or "the stock"
    metrics = {FEATURE_LABELS[key]: value for key, value in values.items()}
    return (
        "You are StockSense AI, a cautious fundamental-analysis assistant for Indian long-term investors. "
        "Use only the supplied fundamentals and do not invent market prices, news, targets, or guarantees. "
        "Return strict JSON with keys recommendation, explanation, description. "
        "recommendation must be one of BUY, WATCHLIST, or AVOID. "
        "explanation must be one concise sentence. description must be 90-140 words with strengths, risks, "
        "and what to verify next. This is educational decision support, not financial advice.\n\n"
        f"Stock: {company}\n"
        f"Score: {score}/100\n"
        f"Metrics: {json.dumps(metrics, sort_keys=True)}\n"
        f"Category scores: {json.dumps(category_scores, sort_keys=True)}\n"
        f"Diagnostics: {json.dumps(diagnostics, sort_keys=True)}"
    )


def _llama_recommendation(values, score, category_scores, diagnostics, stock_name):
    payload = {
        "model": LLM_MODEL,
        "prompt": _build_llm_prompt(values, score, category_scores, diagnostics, stock_name),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.25},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=4) as response:
        outer = json.loads(response.read().decode("utf-8"))
    content = json.loads(outer.get("response", "{}"))
    recommendation = str(content.get("recommendation", "")).upper()
    if recommendation not in {"BUY", "WATCHLIST", "AVOID"}:
        raise ValueError("Llama returned an unsupported recommendation.")
    explanation = str(content.get("explanation", "")).strip()
    description = str(content.get("description", "")).strip()
    if not explanation or not description:
        raise ValueError("Llama response did not include the required explanation fields.")
    return {
        "recommendation": recommendation,
        "tone": _tone_for_recommendation(recommendation),
        "explanation": explanation,
        "description": description,
        "source": f"Ollama {LLM_MODEL}",
    }


def _build_recommendation(values, score, category_scores, diagnostics, stock_name):
    if not LLM_ENABLED:
        return _fallback_recommendation(values, score, category_scores, diagnostics, stock_name)
    try:
        return _llama_recommendation(values, score, category_scores, diagnostics, stock_name)
    except Exception as exc:
        app.logger.info("Llama recommendation unavailable; using local fallback: %s", exc)
        return _fallback_recommendation(values, score, category_scores, diagnostics, stock_name)


def _save_analysis(user_id, stock_name, values, score, recommendation, category_scores, diagnostics):
    get_db().execute(
        """
        INSERT INTO analyses (
            user_id, stock_name, input_values, score, recommendation, explanation,
            category_scores, diagnostics, llm_source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            stock_name or "Unnamed stock",
            json.dumps(values, sort_keys=True),
            score,
            recommendation["recommendation"],
            recommendation["description"],
            json.dumps(category_scores, sort_keys=True),
            json.dumps(diagnostics, sort_keys=True),
            recommendation["source"],
        ),
    )
    get_db().commit()


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


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    if not name or not email or len(password) < 8:
        flash("Enter your name, a valid email, and a password with at least 8 characters.")
        return render_template("signup.html", name=name, email=email), 400
    if _find_user_by_email(email):
        flash("An account with that email already exists. Log in instead.")
        return render_template("login.html", email=email), 400

    cursor = get_db().execute(
        """
        INSERT INTO users (email, name, password_hash, auth_provider)
        VALUES (?, ?, ?, 'password')
        """,
        (email, name, generate_password_hash(password)),
    )
    get_db().commit()
    session.clear()
    session["user_id"] = cursor.lastrowid
    return redirect(url_for("analysis"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", next=request.args.get("next", ""))

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    user = _find_user_by_email(email)
    if user is None or not user["password_hash"] or not check_password_hash(user["password_hash"], password):
        flash("Email or password is incorrect.")
        return render_template("login.html", email=email, next=request.form.get("next", "")), 400

    session.clear()
    session["user_id"] = user["id"]
    next_url = request.form.get("next") or url_for("analysis")
    return redirect(next_url if next_url.startswith("/") and not next_url.startswith("//") else url_for("analysis"))


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.get("/auth/<provider>")
def oauth_login(provider):
    if oauth is None:
        flash("Install Authlib to enable social sign-in.")
        return redirect(url_for("login"))
    if provider not in {"google", "apple"} or provider not in oauth._clients:
        flash(f"{provider.title()} sign-in is not configured yet.")
        return redirect(url_for("login"))
    redirect_uri = url_for("oauth_callback", provider=provider, _external=True)
    return oauth.create_client(provider).authorize_redirect(redirect_uri)


@app.get("/auth/<provider>/callback")
def oauth_callback(provider):
    if oauth is None or provider not in oauth._clients:
        flash("That social sign-in provider is not configured.")
        return redirect(url_for("login"))
    try:
        client = oauth.create_client(provider)
        token = client.authorize_access_token()
        profile = client.parse_id_token(token)
        session.clear()
        session["user_id"] = _create_or_update_oauth_user(provider, dict(profile))
    except Exception as exc:
        app.logger.exception("%s OAuth login failed", provider)
        flash(f"{provider.title()} sign-in failed: {exc}")
        return redirect(url_for("login"))
    return redirect(url_for("analysis"))


@app.get("/history")
@login_required
def history():
    rows = get_db().execute(
        """
        SELECT id, stock_name, score, recommendation, llm_source, created_at
        FROM analyses
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (g.user["id"],),
    ).fetchall()
    return render_template("history.html", analyses=rows)


@app.route("/analysis", methods=["GET", "POST"])
@login_required
def analysis():
    if request.method == "GET":
        return render_template("analysis.html", fields=FIELD_DEFINITIONS, values={})

    stock_name = request.form.get("stock_name", "").strip()
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
                stock_name=stock_name,
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
    diagnostics = _build_stock_diagnostics(values, category_scores)
    recommendation = _build_recommendation(values, score, category_scores, diagnostics, stock_name)
    _save_analysis(g.user["id"], stock_name, values, score, recommendation, category_scores, diagnostics)

    return render_template(
        "result.html",
        stock_name=stock_name,
        score=score,
        recommendation=recommendation["recommendation"],
        tone=recommendation["tone"],
        explanation=recommendation["explanation"],
        recommendation_description=recommendation["description"],
        llm_source=recommendation["source"],
        engine_label=engine_label,
        engine_note=engine_note,
        category_scores=category_scores,
        diagnostics=diagnostics,
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
