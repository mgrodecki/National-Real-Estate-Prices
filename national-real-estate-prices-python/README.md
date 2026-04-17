# National Real Estate Prices (Python)

Python port of [mgrodecki/National-Real-Estate-Prices](https://github.com/mgrodecki/National-Real-Estate-Prices), preserving the original workflow from `Final Project.Rmd`:

- load CPI, mortgage, and housing index datasets
- derive monthly/quarterly inflation-rate series
- generate correlation views and regression charts
- run k-means clustering for the same dataset variants

## Project Layout

- `data/`: source datasets copied from the original repo
- `src/analysis.py`: end-to-end Python analysis pipeline
- `outputs/`: generated CSV summaries and PNG charts

## Notes on CPI Data

The original repository references `CPI-U.csv` but does not include it.

This Python version auto-downloads CPI data from FRED (`CPIAUCSL`) into `data/CPI-U.csv` if that file is missing.

## Refresh Data (Latest Available)

```bash
python src/update_data.py
```

This refreshes:

- `CPI-U.csv` from FRED `CPIAUCSL`
- `30_year_mortgate_rates.csv` from FRED `MORTGAGE30US` (weekly -> monthly average)
- Housing index CSVs from FHFA `hpi_master.csv`

## Run Analysis

```bash
python -m pip install -r requirements.txt
python src/analysis.py
```

## Generate HTML Report

```bash
python src/generate_html_report.py
```

This writes:

- `outputs/up_to_date_analysis_report.html`

## HTML Visualizations

- Open the full dashboard: [outputs/up_to_date_analysis_report.html](outputs/up_to_date_analysis_report.html)

### Preview: Core Trend Charts

![CPI Trend](outputs/line_cpi.png)
![Mortgage Trend](outputs/line_mortgage.png)
![USA HPI Trend](outputs/line_hpi_usa.png)
![Mountain HPI Trend](outputs/line_hpi_mountain.png)

### Preview: Key Relationship Charts

![USA vs Mountain vs CPI Scatter](outputs/scatter_hpi_vs_cpi_usa_mountain.png)
![Colorado/Denver/Boulder vs CPI Scatter](outputs/scatter_hpi_vs_cpi_colorado_denver_boulder.png)
![USA House Inflation vs CPI Inflation](outputs/reg_df_house_usa_vs_cpi.png)
![Mortgage vs CPI Inflation](outputs/reg_df1_mortgage_vs_cpi.png)

### Preview: Correlation and Clustering

![Monthly Correlation Heatmap](outputs/df_monthly_1991_2022_corr_heatmap.png)
![Quarterly Correlation Heatmap](outputs/df2_quarterly_1978_2022_corr_heatmap.png)
![KMeans Monthly k=4](outputs/kmeans_df1_mortgage_cpi_kmeans_k4.png)
![KMeans Inflation k=5](outputs/kmeans_df4_inflation_only_kmeans_k5.png)

After running, review:

- `outputs/regression_summary.csv`
- `outputs/*_corr.csv`
- `outputs/*.png`

## Original Source

- Original R project: `data/Final Project.Rmd`
- Upstream repo: <https://github.com/mgrodecki/National-Real-Estate-Prices>
