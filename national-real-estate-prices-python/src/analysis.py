from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


sns.set_theme(style="whitegrid")


@dataclass
class RegressionResult:
    name: str
    x_col: str
    y_col: str
    slope: float
    intercept: float
    r_value: float
    p_value: float
    std_err: float
    sample_size: int


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_cpi_if_missing(path: Path) -> None:
    if path.exists():
        return

    # Fallback source for monthly CPI if CPI-U.csv is absent in the original repo.
    cpi_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    raw = pd.read_csv(cpi_url)
    if "DATE" in raw.columns:
        date_col = "DATE"
    elif "observation_date" in raw.columns:
        date_col = "observation_date"
    else:
        raise ValueError(f"Unexpected CPI schema columns: {list(raw.columns)}")

    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw.rename(columns={"CPIAUCSL": "Index"})
    cpi = pd.DataFrame(
        {
            "Year": raw[date_col].dt.year,
            "Month": raw[date_col].dt.month,
            "Index": raw["Index"],
        }
    )
    cpi = cpi[(cpi["Year"] >= 1971) & (cpi["Year"] <= 2022)].reset_index(drop=True)
    cpi.to_csv(path, index=False)


def load_data() -> Dict[str, pd.DataFrame]:
    cpi_path = DATA_DIR / "CPI-U.csv"
    download_cpi_if_missing(cpi_path)

    return {
        "CPI": pd.read_csv(cpi_path),
        "Mortgage_Rates": pd.read_csv(DATA_DIR / "30_year_mortgate_rates.csv"),
        "HousingIndex_USA": pd.read_csv(DATA_DIR / "Housing_Index_USA.csv"),
        "HousingIndex_Mountain": pd.read_csv(DATA_DIR / "Housing_Index_Mountain.csv"),
        "HousingIndex_Colorado": pd.read_csv(DATA_DIR / "Housing_Index_Colorado.csv"),
        "HousingIndex_Denver": pd.read_csv(DATA_DIR / "Housing_Index_Denver.csv"),
        "HousingIndex_Boulder": pd.read_csv(DATA_DIR / "Housing_Index_Boulder.csv"),
    }


def add_fractional_year(df: pd.DataFrame, period_col: str, periods_per_year: int) -> pd.DataFrame:
    df = df.copy()
    df["FractionalYear"] = df["Year"] + df[period_col] / periods_per_year
    return df


def save_line_plot(df: pd.DataFrame, x: str, y: str, title: str, filename: str) -> None:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


def save_multi_scatter(
    x: pd.Series,
    ys: List[Tuple[pd.Series, str, str]],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(9, 6))
    for y_values, label, color in ys:
        plt.scatter(x, y_values, alpha=0.75, s=18, label=label, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


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


def regression_with_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, filename: str) -> RegressionResult:
    clean = df[[x_col, y_col]].dropna()
    fit = linregress(clean[x_col], clean[y_col])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=clean, x=x_col, y=y_col, s=24)
    xs = np.linspace(clean[x_col].min(), clean[x_col].max(), 100)
    ys = fit.intercept + fit.slope * xs
    plt.plot(xs, ys, color="red", linewidth=2)
    plt.title(title)
    plt.text(
        0.05,
        0.95,
        f"r = {fit.rvalue:.2f} | p = {fit.pvalue:.3g}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

    return RegressionResult(
        name=title,
        x_col=x_col,
        y_col=y_col,
        slope=fit.slope,
        intercept=fit.intercept,
        r_value=fit.rvalue,
        p_value=fit.pvalue,
        std_err=fit.stderr,
        sample_size=len(clean),
    )


def save_correlation_views(df: pd.DataFrame, cols: List[str], stem: str) -> None:
    corr = df[cols].corr(numeric_only=True)
    corr.to_csv(OUTPUT_DIR / f"{stem}_corr.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Correlation Heatmap: {stem}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{stem}_corr_heatmap.png", dpi=150)
    plt.close()

    pair = sns.pairplot(df[cols].dropna(), diag_kind="hist", corner=True, plot_kws={"s": 14, "alpha": 0.6})
    pair.fig.suptitle(f"Pair Plot: {stem}", y=1.02)
    pair.savefig(OUTPUT_DIR / f"{stem}_pairplot.png", dpi=150)
    plt.close("all")


def run_kmeans(
    df: pd.DataFrame,
    cols: List[str],
    ks: Iterable[int],
    prefix: str,
) -> pd.DataFrame:
    clean = df[cols].dropna().reset_index(drop=True)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(clean)

    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x_scaled)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(x_scaled)

        grouped = clean.copy()
        grouped["cluster"] = labels
        grouped.to_csv(OUTPUT_DIR / f"{prefix}_kmeans_k{k}.csv", index=False)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=labels, palette="tab10", s=28)
        plt.title(f"{prefix}: KMeans k={k} (PCA view)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(title="cluster")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{prefix}_kmeans_k{k}.png", dpi=150)
        plt.close()

    return clean


def main() -> None:
    ensure_dirs()
    data = load_data()

    cpi = add_fractional_year(data["CPI"], "Month", 12)
    mortgage = add_fractional_year(data["Mortgage_Rates"], "Month", 12)
    hi_usa = add_fractional_year(data["HousingIndex_USA"], "Month", 12)
    hi_mtn = add_fractional_year(data["HousingIndex_Mountain"], "Month", 12)
    hi_co = add_fractional_year(data["HousingIndex_Colorado"], "Quarter", 4)
    hi_den = add_fractional_year(data["HousingIndex_Denver"], "Quarter", 4)
    hi_bou = add_fractional_year(data["HousingIndex_Boulder"], "Quarter", 4)

    save_line_plot(cpi, "FractionalYear", "Index", "CPI over Time", "line_cpi.png")
    save_line_plot(mortgage, "FractionalYear", "Rate", "30-Year Fixed Mortgage Rate over Time", "line_mortgage.png")
    save_line_plot(hi_usa, "FractionalYear", "Index", "Housing Price Index USA over Time", "line_hpi_usa.png")
    save_line_plot(hi_mtn, "FractionalYear", "Index", "Housing Price Index Mountain over Time", "line_hpi_mountain.png")
    save_line_plot(hi_co, "FractionalYear", "Index", "Housing Price Index Colorado over Time", "line_hpi_colorado.png")
    save_line_plot(hi_den, "FractionalYear", "Index", "Housing Price Index Denver over Time", "line_hpi_denver.png")
    save_line_plot(hi_bou, "FractionalYear", "Index", "Housing Price Index Boulder over Time", "line_hpi_boulder.png")

    # 1991-2022 monthly scope
    df = cpi.merge(hi_usa, on="FractionalYear", suffixes=("_CPI", "_USA"))
    df = df.merge(hi_mtn, on="FractionalYear", suffixes=("", "_MOUNTAIN"))
    df = df.merge(mortgage, on="FractionalYear", suffixes=("", "_MORT"))
    df = df[["FractionalYear", "Index_CPI", "Index_USA", "Index", "Rate"]].copy()
    df.columns = ["Year", "CPI", "HousingIndex_USA", "HousingIndex_Mountain", "Mortgage_Rate"]

    save_multi_scatter(
        x=df["CPI"],
        ys=[
            (df["HousingIndex_USA"], "Housing Index USA", "blue"),
            (df["HousingIndex_Mountain"], "Housing Index Mountain", "red"),
        ],
        title="Housing Index vs CPI (USA and Mountain)",
        xlabel="Consumer Price Index",
        ylabel="Housing Price Index",
        filename="scatter_hpi_vs_cpi_usa_mountain.png",
    )

    # 1971-2022 monthly CPI + mortgage
    df1 = cpi.merge(mortgage, on="FractionalYear", suffixes=("_CPI", "_MORT"))
    df1 = df1[["FractionalYear", "Index", "Rate"]].copy()
    df1.columns = ["Year", "CPI", "Mortgage_Rate"]

    # 1978-2022 quarterly scope
    df2 = cpi.merge(hi_co, on="FractionalYear", suffixes=("_CPI", "_CO"))
    df2 = df2.merge(hi_den, on="FractionalYear", suffixes=("", "_DEN"))
    df2 = df2.merge(hi_bou, on="FractionalYear", suffixes=("", "_BOU"))
    df2 = df2.merge(mortgage, on="FractionalYear", suffixes=("", "_MORT"))
    df2 = df2[["FractionalYear", "Index_CPI", "Index_CO", "Index", "Index_BOU", "Rate"]].copy()
    df2.columns = ["Year", "CPI", "HousingIndex_Colorado", "HousingIndex_Denver", "HousingIndex_Boulder", "Mortgage_Rate"]

    save_multi_scatter(
        x=df2["CPI"],
        ys=[
            (df2["HousingIndex_Colorado"], "Housing Index Colorado", "red"),
            (df2["HousingIndex_Denver"], "Housing Index Denver", "blue"),
            (df2["HousingIndex_Boulder"], "Housing Index Boulder", "green"),
        ],
        title="Housing Index vs CPI (Colorado / Denver / Boulder)",
        xlabel="Consumer Price Index",
        ylabel="Housing Price Index",
        filename="scatter_hpi_vs_cpi_colorado_denver_boulder.png",
    )

    # Inflation rates
    df["Consumer_Inflation_Rate"] = annualized_inflation(df["CPI"], periods_per_year=12)
    df["House_Inflation_Rate_USA"] = annualized_inflation(df["HousingIndex_USA"], periods_per_year=12)
    df["House_Inflation_Rate_Mountain"] = annualized_inflation(df["HousingIndex_Mountain"], periods_per_year=12)

    df1["Consumer_Inflation_Rate"] = annualized_inflation(df1["CPI"], periods_per_year=12)
    df1 = df1[["Year", "Mortgage_Rate", "Consumer_Inflation_Rate"]]

    df2["Consumer_Inflation_Rate"] = annualized_inflation(df2["CPI"], periods_per_year=4)
    df2["House_Inflation_Rate_Colorado"] = annualized_inflation(df2["HousingIndex_Colorado"], periods_per_year=4)
    df2["House_Inflation_Rate_Denver"] = annualized_inflation(df2["HousingIndex_Denver"], periods_per_year=4)
    df2["House_Inflation_Rate_Boulder"] = annualized_inflation(df2["HousingIndex_Boulder"], periods_per_year=4)

    df3 = df2.merge(df, on="Year", suffixes=("", "_monthly"))

    # Save prepared tables
    df.to_csv(OUTPUT_DIR / "df_monthly_1991_2022.csv", index=False)
    df1.to_csv(OUTPUT_DIR / "df1_monthly_1971_2022.csv", index=False)
    df2.to_csv(OUTPUT_DIR / "df2_quarterly_1978_2022.csv", index=False)
    df3.to_csv(OUTPUT_DIR / "df3_quarterly_merged.csv", index=False)

    # Correlation views
    save_correlation_views(
        df,
        [
            "Consumer_Inflation_Rate",
            "Mortgage_Rate",
            "House_Inflation_Rate_USA",
            "House_Inflation_Rate_Mountain",
        ],
        "df_monthly_1991_2022",
    )
    save_correlation_views(
        df2,
        [
            "Consumer_Inflation_Rate",
            "Mortgage_Rate",
            "House_Inflation_Rate_Colorado",
            "House_Inflation_Rate_Denver",
            "House_Inflation_Rate_Boulder",
        ],
        "df2_quarterly_1978_2022",
    )
    save_correlation_views(
        df3,
        [
            "Consumer_Inflation_Rate",
            "Mortgage_Rate",
            "House_Inflation_Rate_Colorado",
            "House_Inflation_Rate_Denver",
            "House_Inflation_Rate_Boulder",
            "House_Inflation_Rate_USA",
            "House_Inflation_Rate_Mountain",
        ],
        "df3_quarterly_merged",
    )

    regressions: List[RegressionResult] = []

    # Monthly (1991-2022)
    regressions.append(
        regression_with_plot(
            df,
            "Consumer_Inflation_Rate",
            "House_Inflation_Rate_USA",
            "House Inflation Rate (USA) vs Consumer Inflation Rate",
            "reg_df_house_usa_vs_cpi.png",
        )
    )
    regressions.append(
        regression_with_plot(
            df,
            "Consumer_Inflation_Rate",
            "House_Inflation_Rate_Mountain",
            "House Inflation Rate (Mountain) vs Consumer Inflation Rate",
            "reg_df_house_mountain_vs_cpi.png",
        )
    )
    regressions.append(
        regression_with_plot(
            df,
            "House_Inflation_Rate_USA",
            "House_Inflation_Rate_Mountain",
            "House Inflation Rate (Mountain) vs House Inflation Rate (USA)",
            "reg_df_house_mountain_vs_usa.png",
        )
    )
    regressions.append(
        regression_with_plot(
            df,
            "Consumer_Inflation_Rate",
            "Mortgage_Rate",
            "Mortgage Rate vs Consumer Inflation Rate (1991-2022)",
            "reg_df_mortgage_vs_cpi.png",
        )
    )
    regressions.append(
        regression_with_plot(
            df,
            "Mortgage_Rate",
            "House_Inflation_Rate_USA",
            "House Inflation Rate (USA) vs Mortgage Rate (1991-2022)",
            "reg_df_house_usa_vs_mortgage.png",
        )
    )

    # Quarterly (1978-2022)
    regressions.append(
        regression_with_plot(
            df2,
            "Consumer_Inflation_Rate",
            "House_Inflation_Rate_Boulder",
            "House Inflation Rate (Boulder) vs Consumer Inflation Rate",
            "reg_df2_boulder_vs_cpi.png",
        )
    )
    regressions.append(
        regression_with_plot(
            df2,
            "Consumer_Inflation_Rate",
            "House_Inflation_Rate_Colorado",
            "House Inflation Rate (Colorado) vs Consumer Inflation Rate",
            "reg_df2_colorado_vs_cpi.png",
        )
    )
    regressions.append(
        regression_with_plot(
            df2,
            "House_Inflation_Rate_Colorado",
            "House_Inflation_Rate_Boulder",
            "House Inflation Rate (Boulder) vs House Inflation Rate (Colorado)",
            "reg_df2_boulder_vs_colorado.png",
        )
    )
    regressions.append(
        regression_with_plot(
            df2,
            "Consumer_Inflation_Rate",
            "Mortgage_Rate",
            "Mortgage Rate vs Consumer Inflation Rate (1978-2022)",
            "reg_df2_mortgage_vs_cpi.png",
        )
    )
    regressions.append(
        regression_with_plot(
            df2,
            "Mortgage_Rate",
            "House_Inflation_Rate_Boulder",
            "House Inflation Rate (Boulder) vs Mortgage Rate (1978-2022)",
            "reg_df2_boulder_vs_mortgage.png",
        )
    )

    # Monthly (1971-2022)
    regressions.append(
        regression_with_plot(
            df1,
            "Consumer_Inflation_Rate",
            "Mortgage_Rate",
            "Mortgage Rate vs Consumer Inflation Rate (1971-2022)",
            "reg_df1_mortgage_vs_cpi.png",
        )
    )

    reg_df = pd.DataFrame([r.__dict__ for r in regressions])
    reg_df.to_csv(OUTPUT_DIR / "regression_summary.csv", index=False)

    # K-means analyses
    run_kmeans(
        df3,
        [
            "CPI",
            "HousingIndex_Colorado",
            "HousingIndex_Denver",
            "HousingIndex_Boulder",
            "Mortgage_Rate",
            "Consumer_Inflation_Rate",
            "House_Inflation_Rate_Colorado",
            "House_Inflation_Rate_Denver",
            "House_Inflation_Rate_Boulder",
            "House_Inflation_Rate_USA",
            "House_Inflation_Rate_Mountain",
        ],
        ks=[4, 5, 6],
        prefix="kmeans_df3_all_indexes",
    )

    df4_cols = [
        "Year",
        "Mortgage_Rate",
        "Consumer_Inflation_Rate",
        "House_Inflation_Rate_Colorado",
        "House_Inflation_Rate_Denver",
        "House_Inflation_Rate_Boulder",
        "House_Inflation_Rate_USA",
        "House_Inflation_Rate_Mountain",
    ]
    df4 = df3[df4_cols].copy()
    run_kmeans(
        df4,
        [
            "Year",
            "Mortgage_Rate",
            "Consumer_Inflation_Rate",
            "House_Inflation_Rate_Colorado",
            "House_Inflation_Rate_Denver",
            "House_Inflation_Rate_Boulder",
            "House_Inflation_Rate_USA",
            "House_Inflation_Rate_Mountain",
        ],
        ks=[4, 5, 6],
        prefix="kmeans_df4_inflation_only",
    )

    run_kmeans(
        df1,
        ["Year", "Mortgage_Rate", "Consumer_Inflation_Rate"],
        ks=[3, 4, 5, 6],
        prefix="kmeans_df1_mortgage_cpi",
    )

    print(f"Completed. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
