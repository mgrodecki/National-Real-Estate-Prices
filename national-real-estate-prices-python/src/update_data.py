from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
FRED_MORTGAGE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
FHFA_MASTER_URL = "https://www.fhfa.gov/hpi/download/monthly/hpi_master.csv"


def load_fred_series(url: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col]),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
        }
    ).dropna()
    return out


def write_cpi() -> None:
    cpi = load_fred_series(FRED_CPI_URL, "CPIAUCSL")
    cpi = cpi[cpi["date"].dt.year >= 1971].copy()
    cpi_out = pd.DataFrame(
        {
            "Year": cpi["date"].dt.year,
            "Month": cpi["date"].dt.month,
            "Index": cpi["value"].round(3),
        }
    )
    cpi_out.to_csv(DATA_DIR / "CPI-U.csv", index=False)


def write_mortgage_monthly() -> None:
    mortgage = load_fred_series(FRED_MORTGAGE_URL, "MORTGAGE30US")
    mortgage = mortgage[mortgage["date"].dt.year >= 1971].copy()

    monthly = (
        mortgage.assign(Year=mortgage["date"].dt.year, Month=mortgage["date"].dt.month)
        .groupby(["Year", "Month"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "Rate"})
    )
    monthly["Rate"] = monthly["Rate"].round(2)
    monthly.to_csv(DATA_DIR / "30_year_mortgate_rates.csv", index=False)


def select_hpi(
    hpi: pd.DataFrame,
    *,
    hpi_flavor: str,
    frequency: str,
    level: str,
    place_name: str,
) -> pd.DataFrame:
    s = hpi[
        (hpi["hpi_type"] == "traditional")
        & (hpi["hpi_flavor"] == hpi_flavor)
        & (hpi["frequency"] == frequency)
        & (hpi["level"] == level)
        & (hpi["place_name"] == place_name)
    ].copy()

    if s.empty:
        raise ValueError(
            f"No FHFA series found for {hpi_flavor=} {frequency=} {level=} {place_name=}"
        )

    s = s.sort_values(["yr", "period"])
    return s


def write_hpi_files() -> None:
    hpi = pd.read_csv(FHFA_MASTER_URL)

    usa = select_hpi(
        hpi,
        hpi_flavor="purchase-only",
        frequency="monthly",
        level="USA or Census Division",
        place_name="United States",
    )
    mountain = select_hpi(
        hpi,
        hpi_flavor="purchase-only",
        frequency="monthly",
        level="USA or Census Division",
        place_name="Mountain Division",
    )
    colorado = select_hpi(
        hpi,
        hpi_flavor="all-transactions",
        frequency="quarterly",
        level="State",
        place_name="Colorado",
    )
    denver = select_hpi(
        hpi,
        hpi_flavor="all-transactions",
        frequency="quarterly",
        level="MSA",
        place_name="Denver-Aurora-Centennial, CO",
    )
    boulder = select_hpi(
        hpi,
        hpi_flavor="all-transactions",
        frequency="quarterly",
        level="MSA",
        place_name="Boulder, CO",
    )

    pd.DataFrame(
        {"Year": usa["yr"].astype(int), "Month": usa["period"].astype(int), "Index": usa["index_nsa"].round(2)}
    ).to_csv(DATA_DIR / "Housing_Index_USA.csv", index=False)

    pd.DataFrame(
        {
            "Year": mountain["yr"].astype(int),
            "Month": mountain["period"].astype(int),
            "Index": mountain["index_nsa"].round(2),
        }
    ).to_csv(DATA_DIR / "Housing_Index_Mountain.csv", index=False)

    pd.DataFrame(
        {
            "Year": colorado["yr"].astype(int),
            "Quarter": colorado["period"].astype(int),
            "Index": colorado["index_nsa"].round(2),
        }
    ).to_csv(DATA_DIR / "Housing_Index_Colorado.csv", index=False)

    pd.DataFrame(
        {
            "Year": denver["yr"].astype(int),
            "Quarter": denver["period"].astype(int),
            "Index": denver["index_nsa"].round(2),
        }
    ).to_csv(DATA_DIR / "Housing_Index_Denver.csv", index=False)

    pd.DataFrame(
        {
            "Year": boulder["yr"].astype(int),
            "Quarter": boulder["period"].astype(int),
            "Index": boulder["index_nsa"].round(2),
        }
    ).to_csv(DATA_DIR / "Housing_Index_Boulder.csv", index=False)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_cpi()
    write_mortgage_monthly()
    write_hpi_files()
    print(f"Updated datasets in {DATA_DIR}")


if __name__ == "__main__":
    main()
