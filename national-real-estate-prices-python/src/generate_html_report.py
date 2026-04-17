from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_PATH = OUTPUT_DIR / "up_to_date_analysis_report.html"


def latest_point(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    if "Month" in df.columns:
        year = int(df["Year"].max())
        month = int(df[df["Year"] == year]["Month"].max())
        return f"{year}-{month:02d}"
    if "Quarter" in df.columns:
        year = int(df["Year"].max())
        quarter = int(df[df["Year"] == year]["Quarter"].max())
        return f"{year} Q{quarter}"
    return str(df["Year"].max())


def build_coverage_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Dataset": "CPI-U", "Latest Observation": latest_point(DATA_DIR / "CPI-U.csv")},
            {
                "Dataset": "30-Year Mortgage Rate (monthly avg)",
                "Latest Observation": latest_point(DATA_DIR / "30_year_mortgate_rates.csv"),
            },
            {"Dataset": "HPI USA (monthly)", "Latest Observation": latest_point(DATA_DIR / "Housing_Index_USA.csv")},
            {
                "Dataset": "HPI Mountain Division (monthly)",
                "Latest Observation": latest_point(DATA_DIR / "Housing_Index_Mountain.csv"),
            },
            {
                "Dataset": "HPI Colorado (quarterly)",
                "Latest Observation": latest_point(DATA_DIR / "Housing_Index_Colorado.csv"),
            },
            {
                "Dataset": "HPI Denver (quarterly)",
                "Latest Observation": latest_point(DATA_DIR / "Housing_Index_Denver.csv"),
            },
            {
                "Dataset": "HPI Boulder (quarterly)",
                "Latest Observation": latest_point(DATA_DIR / "Housing_Index_Boulder.csv"),
            },
        ]
    )


def img_grid(paths: list[str]) -> str:
    cards = []
    for p in paths:
        cards.append(
            f"""
            <figure class=\"card\">
              <img src=\"{p}\" alt=\"{p}\" loading=\"lazy\" />
              <figcaption>{p}</figcaption>
            </figure>
            """
        )
    return "\n".join(cards)


def main() -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    coverage = build_coverage_table()

    regression = pd.read_csv(OUTPUT_DIR / "regression_summary.csv")
    regression = regression.sort_values("r_value", key=lambda s: s.abs(), ascending=False)
    regression_view = regression[
        ["name", "x_col", "y_col", "sample_size", "slope", "r_value", "p_value"]
    ].rename(
        columns={
            "name": "Model",
            "x_col": "X",
            "y_col": "Y",
            "sample_size": "N",
            "slope": "Slope",
            "r_value": "R",
            "p_value": "P-value",
        }
    )

    overlap = pd.read_csv(OUTPUT_DIR / "overlap_regression_comparison.csv")
    overlap = overlap.sort_values("delta_r", key=lambda s: s.abs(), ascending=False)
    overlap_view = overlap[
        ["frame", "model", "n", "baseline_slope", "updated_slope", "delta_slope", "baseline_r", "updated_r", "delta_r"]
    ].rename(
        columns={
            "frame": "Frame",
            "model": "Model",
            "n": "N",
            "baseline_slope": "Baseline Slope",
            "updated_slope": "Updated Slope",
            "delta_slope": "Delta Slope",
            "baseline_r": "Baseline R",
            "updated_r": "Updated R",
            "delta_r": "Delta R",
        }
    )

    html = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Up-to-Date Real Estate Analysis Report</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #5f6c7b;
      --accent: #14532d;
      --line: #dde4ea;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #e7f1ea 0%, var(--bg) 30%);
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 28px 18px 40px; }}
    h1 {{ margin: 0 0 10px; font-size: 32px; color: #0f3d26; }}
    h2 {{ margin: 30px 0 10px; font-size: 22px; color: #123d2b; }}
    p, li {{ color: var(--muted); line-height: 1.45; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 8px 20px rgba(18, 61, 43, 0.06);
    }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(200px,1fr)); gap: 12px; }}
    .badge {{ background: #edf7ef; border: 1px solid #c9e6d1; border-radius: 10px; padding: 10px 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border: 1px solid var(--line); padding: 8px; text-align: left; }}
    th {{ background: #f1f5f9; }}
    .table-wrap {{ overflow-x: auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }}
    .card {{ margin: 0; background: #fff; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }}
    .card img {{ width: 100%; height: auto; display: block; background: #fff; }}
    .card figcaption {{ padding: 8px 10px; font-size: 12px; color: #4b5563; border-top: 1px solid var(--line); }}
    a {{ color: #0f4f2f; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Up-to-Date National Real Estate Analysis</h1>
    <p>Generated from the refreshed dataset and current pipeline outputs.</p>

    <div class=\"panel meta\">
      <div class=\"badge\"><strong>Generated</strong><br>{generated_at}</div>
      <div class=\"badge\"><strong>Report File</strong><br>{REPORT_PATH.name}</div>
      <div class=\"badge\"><strong>Data Directory</strong><br>{DATA_DIR}</div>
      <div class=\"badge\"><strong>Output Directory</strong><br>{OUTPUT_DIR}</div>
    </div>

    <h2>Data Coverage</h2>
    <div class=\"panel table-wrap\">{coverage.to_html(index=False, border=0, classes='table')}</div>

    <h2>Top Regression Results (Updated)</h2>
    <div class=\"panel table-wrap\">{regression_view.to_html(index=False, border=0, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x), classes='table')}</div>

    <h2>Baseline vs Updated (Overlap Window)</h2>
    <div class=\"panel table-wrap\">{overlap_view.to_html(index=False, border=0, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x), classes='table')}</div>

    <h2>Trend Lines</h2>
    <div class=\"grid\">
      {img_grid(['line_cpi.png','line_mortgage.png','line_hpi_usa.png','line_hpi_mountain.png','line_hpi_colorado.png','line_hpi_denver.png','line_hpi_boulder.png'])}
    </div>

    <h2>Key Scatter & Regression Charts</h2>
    <div class=\"grid\">
      {img_grid([
        'scatter_hpi_vs_cpi_usa_mountain.png',
        'scatter_hpi_vs_cpi_colorado_denver_boulder.png',
        'reg_df_house_usa_vs_cpi.png',
        'reg_df_house_mountain_vs_cpi.png',
        'reg_df_house_mountain_vs_usa.png',
        'reg_df_mortgage_vs_cpi.png',
        'reg_df_house_usa_vs_mortgage.png',
        'reg_df1_mortgage_vs_cpi.png',
        'reg_df2_boulder_vs_cpi.png',
        'reg_df2_colorado_vs_cpi.png',
        'reg_df2_boulder_vs_colorado.png',
        'reg_df2_mortgage_vs_cpi.png',
        'reg_df2_boulder_vs_mortgage.png'
      ])}
    </div>

    <h2>Correlation Visuals</h2>
    <div class=\"grid\">
      {img_grid([
        'df_monthly_1991_2022_corr_heatmap.png',
        'df_monthly_1991_2022_pairplot.png',
        'df2_quarterly_1978_2022_corr_heatmap.png',
        'df2_quarterly_1978_2022_pairplot.png',
        'df3_quarterly_merged_corr_heatmap.png',
        'df3_quarterly_merged_pairplot.png'
      ])}
    </div>

    <h2>K-Means Cluster Views</h2>
    <div class=\"grid\">
      {img_grid([
        'kmeans_df1_mortgage_cpi_kmeans_k3.png',
        'kmeans_df1_mortgage_cpi_kmeans_k4.png',
        'kmeans_df1_mortgage_cpi_kmeans_k5.png',
        'kmeans_df1_mortgage_cpi_kmeans_k6.png',
        'kmeans_df3_all_indexes_kmeans_k4.png',
        'kmeans_df3_all_indexes_kmeans_k5.png',
        'kmeans_df3_all_indexes_kmeans_k6.png',
        'kmeans_df4_inflation_only_kmeans_k4.png',
        'kmeans_df4_inflation_only_kmeans_k5.png',
        'kmeans_df4_inflation_only_kmeans_k6.png'
      ])}
    </div>

    <h2>Raw Output Files</h2>
    <div class=\"panel\">
      <ul>
        <li><a href=\"regression_summary.csv\">regression_summary.csv</a></li>
        <li><a href=\"overlap_regression_comparison.csv\">overlap_regression_comparison.csv</a></li>
        <li><a href=\"overlap_series_delta_summary.csv\">overlap_series_delta_summary.csv</a></li>
        <li><a href=\"overlap_correlation_delta.csv\">overlap_correlation_delta.csv</a></li>
        <li><a href=\"overlap_window_summary.csv\">overlap_window_summary.csv</a></li>
      </ul>
    </div>
  </div>
</body>
</html>
"""

    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
