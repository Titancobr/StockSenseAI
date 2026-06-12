from pathlib import Path

import numpy as np
import pandas as pd


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


def bounded(value, low, high, invert=False):
    score = (value - low) / (high - low) * 100
    if invert:
        score = 100 - score
    return np.clip(score, 0, 100)


def compute_score(frame, rng):
    valuation_gap = frame["company_pe"] / frame["industry_pe"]
    score = (
        0.11 * bounded(valuation_gap, 0.45, 1.35, invert=True)
        + 0.05 * bounded(frame["pb_ratio"], 1.0, 8.0, invert=True)
        + 0.13 * bounded(frame["roe"], 8, 30)
        + 0.17 * bounded(frame["roce"], 10, 38)
        + 0.12 * bounded(frame["debt_to_equity"], 0, 1.7, invert=True)
        + 0.08 * bounded(frame["interest_coverage"], 2, 18)
        + 0.08 * bounded(frame["sales_growth_5y"], 0, 28)
        + 0.08 * bounded(frame["profit_growth_5y"], 0, 35)
        + 0.05 * bounded(frame["operating_profit_margin"], 8, 32)
        + 0.05 * bounded(frame["free_cash_flow"], -500, 7000)
        + 0.04 * bounded(frame["promoter_holding"], 25, 75)
        + 0.05 * bounded(frame["promoter_pledge"], 0, 20, invert=True)
        + 0.04 * bounded(frame["current_ratio"], 0.8, 2.5)
        + 0.07 * bounded(frame["eps_growth"], 0, 32)
    )

    red_flags = (
        (frame["debt_to_equity"] > 2.0) * 8
        + (frame["debt_to_equity"] > 3.5) * 8
        + (frame["interest_coverage"] < 2.0) * 10
        + (frame["interest_coverage"] < 1.0) * 8
        + (frame["promoter_pledge"] > 20) * 12
        + (frame["promoter_pledge"] > 45) * 10
        + (frame["free_cash_flow"] < -1500) * 7
        + (frame["free_cash_flow"] < -5000) * 8
        + (frame["roe"] < 5) * 6
        + (frame["roce"] < 6) * 6
        + (frame["current_ratio"] < 0.75) * 5
        + (valuation_gap > 1.8) * 8
    )

    quality_bonus = (
        (frame["roce"] > 28)
        & (frame["roe"] > 20)
        & (frame["debt_to_equity"] < 0.5)
        & (frame["promoter_pledge"] < 2)
        & (frame["free_cash_flow"] > 2500)
    ) * 6

    return np.clip(score - red_flags + quality_bonus + rng.normal(0, 3.0, len(frame)), 0, 100)


def add_edge_cases(rng):
    rows = []
    archetypes = [
        ("quality_compounder", 96, [18, 28, 6, 32, 42, 0.05, 55, 22, 28, 29, 16000, 68, 0, 2.7, 26]),
        ("overvalued_quality", 74, [85, 35, 18, 31, 39, 0.02, 70, 24, 26, 31, 13000, 72, 0, 2.8, 24]),
        ("cheap_turnaround", 63, [8, 22, 0.9, 11, 13, 0.75, 5, 8, 18, 12, 1200, 44, 3, 1.25, 15]),
        ("debt_trap", 18, [14, 24, 1.1, 7, 6, 3.8, 0.8, 5, -12, 8, -3500, 39, 18, 0.65, -15]),
        ("pledge_risk", 35, [16, 25, 2.2, 16, 18, 0.6, 7, 13, 12, 17, 2200, 58, 55, 1.4, 11]),
        ("negative_cashflow_growth", 48, [28, 32, 5.5, 21, 25, 0.4, 12, 32, 38, 15, -6500, 49, 0, 1.1, 36]),
        ("deep_value_slow", 58, [7, 19, 0.7, 13, 15, 0.2, 18, 2, 4, 14, 1900, 51, 0, 2.1, 3]),
        ("cyclical_peak", 52, [5, 11, 1.5, 35, 46, 0.9, 10, 40, 80, 38, 8000, 46, 2, 1.3, 75]),
        ("loss_maker", 8, [0, 18, 7, -12, -9, 1.8, 0.5, -18, -40, -8, -8000, 21, 25, 0.5, -35]),
        ("zero_debt_missing_like", 82, [24, 31, 4.2, 24, 31, 0, 100, 16, 18, 23, 5200, 63, 0, 2.2, 17]),
    ]
    for i, (name, base_score, values) in enumerate(archetypes):
        for copy in range(250):
            noise = rng.normal(0, [3, 3, 0.5, 2, 2.5, 0.12, 3, 3, 4, 2, 800, 4, 2, 0.18, 4])
            adjusted = np.array(values, dtype=float) + noise
            rows.append([f"{name}_{copy:03d}", 2020 + (i + copy) % 5, *adjusted, np.clip(base_score + rng.normal(0, 4), 0, 100)])
    return pd.DataFrame(rows, columns=["company", "year", *FEATURE_COLUMNS, "score"])


def generate_dataset(rows=60000, seed=42):
    rng = np.random.default_rng(seed)

    regimes = rng.choice(
        ["quality", "average", "cyclical", "distressed", "high_growth", "expensive"],
        size=rows,
        p=[0.22, 0.30, 0.14, 0.12, 0.12, 0.10],
    )
    sector_quality = rng.beta(2.8, 2.2, rows)
    valuation_pressure = rng.beta(2.0, 2.4, rows)
    leverage_risk = rng.beta(1.8, 4.2, rows)

    sector_quality = np.where(regimes == "quality", np.maximum(sector_quality, rng.uniform(0.65, 0.95, rows)), sector_quality)
    sector_quality = np.where(regimes == "distressed", np.minimum(sector_quality, rng.uniform(0.05, 0.35, rows)), sector_quality)
    sector_quality = np.where(regimes == "high_growth", np.maximum(sector_quality, rng.uniform(0.55, 0.90, rows)), sector_quality)
    valuation_pressure = np.where(regimes == "expensive", np.maximum(valuation_pressure, rng.uniform(0.75, 1.0, rows)), valuation_pressure)
    leverage_risk = np.where(regimes == "distressed", np.maximum(leverage_risk, rng.uniform(0.65, 1.0, rows)), leverage_risk)

    industry_pe = rng.normal(18 + sector_quality * 28, 5).clip(6, 75)
    company_pe = (industry_pe * rng.normal(0.72 + valuation_pressure * 0.85, 0.20)).clip(3, 110)
    pb_ratio = rng.lognormal(0.35 + sector_quality * 1.2, 0.45).clip(0.25, 25)
    roe = rng.normal(5 + sector_quality * 28, 6).clip(-18, 55)
    roce = (roe + rng.normal(2 + sector_quality * 9, 5)).clip(-15, 65)
    debt_to_equity = rng.lognormal(-1.2 + leverage_risk * 1.7, 0.55).clip(0, 5)
    interest_coverage = rng.lognormal(1.1 + sector_quality * 2.1 - leverage_risk * 1.2, 0.75).clip(0.2, 100)
    sales_growth_5y = rng.normal(-2 + sector_quality * 28, 8).clip(-35, 70)
    sales_growth_5y = np.where(regimes == "high_growth", sales_growth_5y + rng.uniform(10, 28, rows), sales_growth_5y).clip(-35, 95)
    sales_growth_5y = np.where(regimes == "cyclical", sales_growth_5y + rng.normal(0, 18, rows), sales_growth_5y).clip(-45, 110)
    profit_growth_5y = (sales_growth_5y + rng.normal(-1 + sector_quality * 8, 8)).clip(-55, 95)
    operating_profit_margin = rng.normal(5 + sector_quality * 28, 7).clip(-12, 60)
    free_cash_flow = rng.normal(-700 + sector_quality * 8500 - leverage_risk * 1200, 2200).clip(-9000, 45000)
    promoter_holding = rng.normal(32 + sector_quality * 42, 13).clip(0, 90)
    promoter_pledge = rng.lognormal(0.2 + leverage_risk * 2.2 - sector_quality * 1.0, 0.75).clip(0, 85)
    current_ratio = rng.normal(0.75 + sector_quality * 1.8 - leverage_risk * 0.35, 0.45).clip(0.15, 5)
    eps_growth = (profit_growth_5y + rng.normal(0, 8)).clip(-60, 100)

    frame = pd.DataFrame(
        {
            "company": [f"SYNTH{i:05d}" for i in range(rows)],
            "year": rng.integers(2016, 2025, rows),
            "company_pe": company_pe.round(2),
            "industry_pe": industry_pe.round(2),
            "pb_ratio": pb_ratio.round(2),
            "roe": roe.round(2),
            "roce": roce.round(2),
            "debt_to_equity": debt_to_equity.round(2),
            "interest_coverage": interest_coverage.round(2),
            "sales_growth_5y": sales_growth_5y.round(2),
            "profit_growth_5y": profit_growth_5y.round(2),
            "operating_profit_margin": operating_profit_margin.round(2),
            "free_cash_flow": free_cash_flow.round(2),
            "promoter_holding": promoter_holding.round(2),
            "promoter_pledge": promoter_pledge.round(2),
            "current_ratio": current_ratio.round(2),
            "eps_growth": eps_growth.round(2),
        }
    )
    frame["score"] = compute_score(frame, rng).round(2)
    edge_cases = add_edge_cases(rng)
    full = pd.concat([frame, edge_cases], ignore_index=True)
    for column in FEATURE_COLUMNS + ["score"]:
        full[column] = full[column].round(2)
    return full.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    output_path = Path("data/training_data.csv")
    output_path.parent.mkdir(exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(output_path, index=False)
    print(f"Saved {len(dataset):,} rows to {output_path}")
