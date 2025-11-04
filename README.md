  Public Transport Passenger Journey Forecasting

   Project Overview

This repository contains a complete forecasting pipeline for public transport passenger journeys using SARIMA (Seasonal AutoRegressive Integrated Moving Average) models. The project includes data cleaning, exploratory data analysis, outlier detection, and 7-day forecasting.

   Repository Structure

```
.
├── Forecast_Analysis.ipynb             -Main Jupyter notebook with all implementation steps
├── INSIGHTS_REPORT.md                  -Key insights and actionable findings
├── PROCEDURES.md                       -Methodology, procedures, and technical details
├── README.md                           -This file
├── Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv    -Raw data
├── cleaned_daily_by_service.csv       -Cleaned time series data
├── 7_day_forecast_sarima.csv          -Complete 7-day forecast
├── forecast_7days_ .csv               -Individual service forecasts
└──  .png                               -Visualization outputs
```

   Quick Start

  Prerequisites

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter
```

  Running the Analysis

1.   Open Jupyter Notebook  :
   ```bash
   jupyter notebook Forecast_Analysis.ipynb
   ```

2.   Run all cells   sequentially to execute the complete analysis pipeline

3.   Check outputs  :
   - Forecasts: `7_day_forecast_sarima.csv` and `forecast_7days_ .csv`
    - Forecasts: `7_day_forecast_sarima.csv` (wide) and `forecast_7days_all.csv` (combined long format)
   - Visualizations: ` .png` files
   - Insights: See `INSIGHTS_REPORT.md`

   Documentation

-   INSIGHTS_REPORT.md  : Key findings, patterns, and actionable recommendations
-   PROCEDURES.md  : Detailed methodology, assumptions, limitations, and procedures
-   Forecast_Analysis.ipynb  : Complete implementation with all code steps

   Key Features

- ✅ Comprehensive data cleaning and missing data handling
- ✅ Outlier detection with visualizations
- ✅ Exploratory data analysis with actionable insights
- ✅ SARIMA forecasting with confidence intervals
- ✅ Residual diagnostics and model validation
- ✅ 7-day forecasts for 5 service types

How to run (script)
-------------------
If you prefer running the script directly (no notebook):
1. Create a Python environment with the listed packages.
2. From the project root run:

```powershell
python Forecast.py
```

Outputs created by the script:
- `cleaned_daily_by_service.csv` — cleaned series
- `outliers_detailed_visualization.png`, `outliers_boxplots.png`, `actionable_insights.png`
- `7_day_forecast_sarima.csv` — wide-format forecast
- `forecast_7days_all.csv` — combined per-service forecast (Service, Date, Forecast, Lower_CI, Upper_CI)
- `7_day_forecast_sarima_visualization.png`

That is all you need to reproduce the forecasts.

   Output Files

1.   Forecasts  :
   - `7_day_forecast_sarima.csv`: Combined forecast with confidence intervals
   - `forecast_7days_ .csv`: Individual service forecasts

2.   Data  :
   - `cleaned_daily_by_service.csv`: Cleaned continuous daily time series

3.   Visualizations  :
   - `outliers_detailed_visualization.png`: Outliers highlighted in time series
   - `outliers_boxplots.png`: Box plots for outlier detection
   - `actionable_insights.png`: Key insights visualization
   - `7_day_forecast_sarima_visualization.png`: Forecast vs historical comparison

   Service Types Forecasted

1. Local Route
2. Light Rail
3. Peak Service
4. Rapid Route
5. School

   Key Insights Summary

-   Rapid Route   handles 39% of total passenger volume
-   Weekends   show 50-80% fewer passengers than weekdays
-   Sunday   is the least crowded day (83% fewer passengers vs Wednesday)
-   February   shows peak monthly volumes
-   COVID-19   impact: 42% decrease during 2020-2021

   Methodology

-   Algorithm  : SARIMA with weekly seasonality (7-day period)
-   Validation  : Temporal split (last 30 days held out)
-   Outlier Detection  : IQR method (1.5× threshold)
-   Missing Data  : Time-based interpolation + forward/backward fill


