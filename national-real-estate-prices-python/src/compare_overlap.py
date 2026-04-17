from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress


BASE = Path(__file__).resolve().parents[1]
UPDATED_DATA = BASE / "data"
BASELINE_DATA = Path(r"C:\Codex\National-Real-Estate-Prices-main")
OUT = BASE / "outputs"


def add_fractional_year(df: pd.DataFrame, period_col: str, periods_per_year: int) -> pd.DataFrame:
    out = df.copy()
    out["FractionalYear"] = out["Year"] + out[period_col] / periods_per_year
    return out


def annualized_inflation(series: pd.Series, periods_per_year: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(n):
        if i < periods_per_year:
            if i + 1 < n:
                out[i] = periods_per_year * (values[i + 1] / values[i] - 1.0) * 100.0
        else:
            out[i] = (values[i] / values[i - periods_per_year] - 1.0) * 100.0
    return pd.Series(out, index=series.index)


def prepare_frames(data_dir: Path, cpi: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    mortgage = pd.read_csv(data_dir / "30_year_mortgate_rates.csv")
    hi_usa = pd.read_csv(data_dir / "Housing_Index_USA.csv")
    hi_mtn = pd.read_csv(data_dir / "Housing_Index_Mountain.csv")
    hi_co = pd.read_csv(data_dir / "Housing_Index_Colorado.csv")
    hi_den = pd.read_csv(data_dir / "Housing_Index_Denver.csv")
    hi_bou = pd.read_csv(data_dir / "Housing_Index_Boulder.csv")

    cpi = add_fractional_year(cpi, "Month", 12)
    mortgage = add_fractional_year(mortgage, "Month", 12)
    hi_usa = add_fractional_year(hi_usa, "Month", 12)
    hi_mtn = add_fractional_year(hi_mtn, "Month", 12)
    hi_co = add_fractional_year(hi_co, "Quarter", 4)
    hi_den = add_fractional_year(hi_den, "Quarter", 4)
    hi_bou = add_fractional_year(hi_bou, "Quarter", 4)

    df = cpi.merge(hi_usa, on="FractionalYear", suffixes=("_CPI", "_USA"))
    df = df.merge(hi_mtn, on="FractionalYear", suffixes=("", "_MOUNTAIN"))
    df = df.merge(mortgage, on="FractionalYear", suffixes=("", "_MORT"))
    df = df[["FractionalYear", "Index_CPI", "Index_USA", "Index", "Rate"]].copy()
    df.columns = ["Year", "CPI", "HousingIndex_USA", "HousingIndex_Mountain", "Mortgage_Rate"]

    df1 = cpi.merge(mortgage, on="FractionalYear", suffixes=("_CPI", "_MORT"))
    df1 = df1[["FractionalYear", "Index", "Rate"]].copy()
    df1.columns = ["Year", "CPI", "Mortgage_Rate"]

    df2 = cpi.merge(hi_co, on="FractionalYear", suffixes=("_CPI", "_CO"))
    df2 = df2.merge(hi_den, on="FractionalYear", suffixes=("", "_DEN"))
    df2 = df2.merge(hi_bou, on="FractionalYear", suffixes=("", "_BOU"))
    df2 = df2.merge(mortgage, on="FractionalYear", suffixes=("", "_MORT"))
    df2 = df2[["FractionalYear", "Index_CPI", "Index_CO", "Index", "Index_BOU", "Rate"]].copy()
    df2.columns = ["Year", "CPI", "HousingIndex_Colorado", "HousingIndex_Denver", "HousingIndex_Boulder", "Mortgage_Rate"]

    df["Consumer_Inflation_Rate"] = annualized_inflation(df["CPI"], 12)
    df["House_Inflation_Rate_USA"] = annualized_inflation(df["HousingIndex_USA"], 12)
    df["House_Inflation_Rate_Mountain"] = annualized_inflation(df["HousingIndex_Mountain"], 12)

    df1["Consumer_Inflation_Rate"] = annualized_inflation(df1["CPI"], 12)

    df2["Consumer_Inflation_Rate"] = annualized_inflation(df2["CPI"], 4)
    df2["House_Inflation_Rate_Colorado"] = annualized_inflation(df2["HousingIndex_Colorado"], 4)
    df2["House_Inflation_Rate_Denver"] = annualized_inflation(df2["HousingIndex_Denver"], 4)
    df2["House_Inflation_Rate_Boulder"] = annualized_inflation(df2["HousingIndex_Boulder"], 4)

    return {"df": df, "df1": df1, "df2": df2}


def overlap(base: pd.DataFrame, up: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = set(base["Year"].round(6)).intersection(set(up["Year"].round(6)))
    b = base[base["Year"].round(6).isin(common)].copy().sort_values("Year").reset_index(drop=True)
    u = up[up["Year"].round(6).isin(common)].copy().sort_values("Year").reset_index(drop=True)
    return b, u


def reg(df: pd.DataFrame, x: str, y: str) -> Dict[str, float]:
    clean = df[[x, y]].dropna()
    fit = linregress(clean[x], clean[y])
    return {"n": len(clean), "slope": fit.slope, "intercept": fit.intercept, "r": fit.rvalue, "p": fit.pvalue}


def compare_regs(base: pd.DataFrame, up: pd.DataFrame, models: List[Tuple[str, str, str]], frame_name: str) -> pd.DataFrame:
    rows = []
    for name, x, y in models:
        br = reg(base, x, y)
        ur = reg(up, x, y)
        rows.append(
            {
                "frame": frame_name,
                "model": name,
                "n": int(br["n"]),
                "baseline_slope": br["slope"],
                "updated_slope": ur["slope"],
                "delta_slope": ur["slope"] - br["slope"],
                "baseline_r": br["r"],
                "updated_r": ur["r"],
                "delta_r": ur["r"] - br["r"],
                "baseline_p": br["p"],
                "updated_p": ur["p"],
            }
        )
    return pd.DataFrame(rows)


def compare_corr(base: pd.DataFrame, up: pd.DataFrame, cols: List[str], frame_name: str) -> pd.DataFrame:
    bc = base[cols].corr(numeric_only=True)
    uc = up[cols].corr(numeric_only=True)
    out = (uc - bc).stack().reset_index()
    out.columns = ["var1", "var2", "delta_corr"]
    out.insert(0, "frame", frame_name)
    return out


def summarize_series_diff(base: pd.DataFrame, up: pd.DataFrame, cols: List[str], frame_name: str) -> pd.DataFrame:
    rows = []
    merged = base[["Year"] + cols].merge(up[["Year"] + cols], on="Year", suffixes=("_baseline", "_updated"))
    for c in cols:
        diff = merged[f"{c}_updated"] - merged[f"{c}_baseline"]
        rows.append(
            {
                "frame": frame_name,
                "series": c,
                "n": int(diff.notna().sum()),
                "mean_delta": float(diff.mean()),
                "median_delta": float(diff.median()),
                "max_abs_delta": float(diff.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    cpi_full = pd.read_csv(UPDATED_DATA / "CPI-U.csv")
    cpi_baseline = cpi_full[cpi_full["Year"] <= 2022].copy()

    baseline = prepare_frames(BASELINE_DATA, cpi_baseline)
    updated = prepare_frames(UPDATED_DATA, cpi_full)

    comparisons = []
    reg_tables = []
    corr_tables = []

    specs = {
        "df": {
            "series": ["CPI", "HousingIndex_USA", "HousingIndex_Mountain", "Mortgage_Rate", "Consumer_Inflation_Rate", "House_Inflation_Rate_USA", "House_Inflation_Rate_Mountain"],
            "corr": ["Consumer_Inflation_Rate", "Mortgage_Rate", "House_Inflation_Rate_USA", "House_Inflation_Rate_Mountain"],
            "models": [
                ("House_USA_vs_CPIInfl", "Consumer_Inflation_Rate", "House_Inflation_Rate_USA"),
                ("House_Mountain_vs_CPIInfl", "Consumer_Inflation_Rate", "House_Inflation_Rate_Mountain"),
                ("House_Mountain_vs_House_USA", "House_Inflation_Rate_USA", "House_Inflation_Rate_Mountain"),
                ("Mortgage_vs_CPIInfl", "Consumer_Inflation_Rate", "Mortgage_Rate"),
                ("House_USA_vs_Mortgage", "Mortgage_Rate", "House_Inflation_Rate_USA"),
            ],
        },
        "df1": {
            "series": ["CPI", "Mortgage_Rate", "Consumer_Inflation_Rate"],
            "corr": ["Consumer_Inflation_Rate", "Mortgage_Rate"],
            "models": [("Mortgage_vs_CPIInfl", "Consumer_Inflation_Rate", "Mortgage_Rate")],
        },
        "df2": {
            "series": ["CPI", "HousingIndex_Colorado", "HousingIndex_Denver", "HousingIndex_Boulder", "Mortgage_Rate", "Consumer_Inflation_Rate", "House_Inflation_Rate_Colorado", "House_Inflation_Rate_Denver", "House_Inflation_Rate_Boulder"],
            "corr": ["Consumer_Inflation_Rate", "Mortgage_Rate", "House_Inflation_Rate_Colorado", "House_Inflation_Rate_Denver", "House_Inflation_Rate_Boulder"],
            "models": [
                ("House_Boulder_vs_CPIInfl", "Consumer_Inflation_Rate", "House_Inflation_Rate_Boulder"),
                ("House_Colorado_vs_CPIInfl", "Consumer_Inflation_Rate", "House_Inflation_Rate_Colorado"),
                ("House_Boulder_vs_House_Colorado", "House_Inflation_Rate_Colorado", "House_Inflation_Rate_Boulder"),
                ("Mortgage_vs_CPIInfl", "Consumer_Inflation_Rate", "Mortgage_Rate"),
                ("House_Boulder_vs_Mortgage", "Mortgage_Rate", "House_Inflation_Rate_Boulder"),
            ],
        },
    }

    overlap_meta = []
    for frame, cfg in specs.items():
        b, u = overlap(baseline[frame], updated[frame])
        overlap_meta.append(
            {
                "frame": frame,
                "overlap_start": float(b["Year"].min()),
                "overlap_end": float(b["Year"].max()),
                "overlap_rows": int(len(b)),
            }
        )

        comparisons.append(summarize_series_diff(b, u, cfg["series"], frame))
        reg_tables.append(compare_regs(b, u, cfg["models"], frame))
        corr_tables.append(compare_corr(b, u, cfg["corr"], frame))

    overlap_df = pd.DataFrame(overlap_meta)
    series_df = pd.concat(comparisons, ignore_index=True)
    reg_df = pd.concat(reg_tables, ignore_index=True)
    corr_df = pd.concat(corr_tables, ignore_index=True)

    overlap_df.to_csv(OUT / "overlap_window_summary.csv", index=False)
    series_df.to_csv(OUT / "overlap_series_delta_summary.csv", index=False)
    reg_df.to_csv(OUT / "overlap_regression_comparison.csv", index=False)
    corr_df.to_csv(OUT / "overlap_correlation_delta.csv", index=False)

    key = reg_df[["frame", "model", "n", "baseline_slope", "updated_slope", "delta_slope", "baseline_r", "updated_r", "delta_r"]]
    print("Overlap windows:")
    print(overlap_df.to_string(index=False))
    print("\nRegression deltas (overlap only):")
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
