import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

print("="*80)
print("STEP 1: DATA LOADING AND INITIAL CLEANING")
print("="*80)

df = pd.read_csv('Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv')

print(f"\nOriginal dataset shape: {df.shape}")
print(f"\nDate range in raw data: {df['Date'].min()} to {df['Date'].max()}")

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

invalid_dates = df[df['Date'].isna()]
if len(invalid_dates) > 0:
    print(f"\nWarning: {len(invalid_dates)} rows with invalid dates removed")
    df = df.dropna(subset=['Date'])

df = df.sort_values('Date').reset_index(drop=True)

service_cols = ['Local Route', 'Light Rail', 'Peak Service', 'Rapid Route', 'School', 'Other']

for col in service_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df.loc[df[col] < 0, col] = np.nan

print(f"\nAfter initial cleaning: {df.shape}")
print(f"\nMissing values per column:")
print(df[service_cols].isnull().sum())

print("\n" + "="*80)
print("STEP 2: CREATING CONTINUOUS DAILY TIME SERIES")
print("="*80)

date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
print(f"\nCreating continuous daily series from {date_range[0]} to {date_range[-1]}")
print(f"Total days in continuous series: {len(date_range)}")

df_ts = df.set_index('Date')[service_cols].reindex(date_range)

print(f"\nMissing days after reindexing:")
missing_days = df_ts.isnull().sum()
print(missing_days)

print("\n" + "="*80)
print("STEP 3: HANDLING MISSING DATA")
print("="*80)

df_clean = df_ts.copy()
for col in service_cols:
    original_missing = df_clean[col].isnull().sum()
    series = df_clean[col]
    weekend_mask = df_clean.index.dayofweek >= 5
    try:
        weekend_zero_ratio = ((series == 0) & weekend_mask).sum() / max(1, weekend_mask.sum())
    except Exception:
        weekend_zero_ratio = 0

    is_zero = series == 0
    zeros_to_impute = pd.Series(False, index=series.index)
    if weekend_zero_ratio < 0.75:
        neigh_nonzero = (series.shift(1) > 0) | (series.shift(-1) > 0)
        zeros_to_impute = is_zero & neigh_nonzero

        grp = (~is_zero).cumsum()
        zero_run_lengths = is_zero.groupby(grp).transform('sum')
        zeros_to_impute = zeros_to_impute | (is_zero & (zero_run_lengths <= 7))

    df_clean.loc[zeros_to_impute, col] = np.nan

    df_clean[col] = df_clean[col].interpolate(method='time', limit=7, limit_direction='both')
    df_clean[col] = df_clean[col].ffill().bfill()

    final_missing = df_clean[col].isnull().sum()

    if original_missing > 0 or zeros_to_impute.sum() > 0:
        print(f"\n{col}:")
        print(f"  Original missing: {original_missing} days ({original_missing/len(df_clean)*100:.1f}%)")
        print(f"  Zeros marked for imputation: {zeros_to_impute.sum()}")
        print(f"  After cleaning: {final_missing} days ({final_missing/len(df_clean)*100:.1f}%)")
        print(f"  Filled: {original_missing + zeros_to_impute.sum() - final_missing} days")

df_clean = df_clean.clip(lower=0)

df_clean.reset_index().rename(columns={'index': 'Date'}).to_csv('cleaned_daily_by_service.csv', index=False)
print(f"\n[OK] Cleaned dataset saved to: cleaned_daily_by_service.csv")

df_clean['Year'] = df_clean.index.year
df_clean['Month'] = df_clean.index.month
df_clean['DayOfWeek'] = df_clean.index.dayofweek
df_clean['DayName'] = df_clean.index.day_name()
df_clean['IsWeekend'] = (df_clean['DayOfWeek'] >= 5).astype(int)

print("\n" + "="*80)
print("STEP 4: EXPLICIT OUTLIER DETECTION AND VISUALIZATION")
print("="*80)

outliers_detailed = {}
outliers_summary = {}

for col in service_cols:
    print(f"\n{'='*60}")
    print(f"OUTLIER ANALYSIS: {col}")
    print('='*60)
    
    series = df_clean[col].dropna()
    
    # IQR Method
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (series < lower_bound) | (series > upper_bound)
    outliers = series[outliers_mask]
    
    outliers_info = pd.DataFrame({
        'Date': outliers.index,
        'Value': outliers.values,
        'Type': ['Low' if v < lower_bound else 'High' for v in outliers.values],
        'Deviation_from_Median': outliers.values - series.median(),
        'Z_Score': (outliers.values - series.mean()) / series.std()
    })
    
    outliers_detailed[col] = outliers_info
    outliers_summary[col] = {
        'count': len(outliers),
        'percent': len(outliers) / len(series) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'high_outliers': len(outliers[outliers > upper_bound]),
        'low_outliers': len(outliers[outliers < lower_bound])
    }
    
    print(f"\nIQR Statistics:")
    print(f"  Q1: {Q1:.2f}")
    print(f"  Q3: {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Lower bound: {lower_bound:.2f}")
    print(f"  Upper bound: {upper_bound:.2f}")
    print(f"\nOutlier Count:")
    print(f"  Total outliers: {len(outliers)} ({len(outliers)/len(series)*100:.2f}%)")
    print(f"  High outliers: {outliers_summary[col]['high_outliers']}")
    print(f"  Low outliers: {outliers_summary[col]['low_outliers']}")
    
    if len(outliers) > 0:
        print(f"\nTop 10 Outliers (by absolute deviation):")
        top_outliers = outliers_info.reindex(
            outliers_info['Deviation_from_Median'].abs().nlargest(10).index
        )
        print(top_outliers[['Date', 'Value', 'Type', 'Deviation_from_Median', 'Z_Score']].to_string(index=False))
    
    # --- Handle outliers by winsorizing to the IQR bounds (but never below 0) ---
    lower_clip = max(0, outliers_summary[col]['lower_bound'])
    upper_clip = outliers_summary[col]['upper_bound']
    # Count values to be clipped for reporting
    to_clip = ((df_clean[col] < lower_clip) | (df_clean[col] > upper_clip)).sum()
    if to_clip > 0:
        print(f"\n{col}: Winsorizing {to_clip} values to bounds [{lower_clip:.0f}, {upper_clip:.0f}]")
    df_clean[col] = df_clean[col].clip(lower=lower_clip, upper=upper_clip)

# Outlier plotting removed by request. Outlier detection still performed above and values were winsorized.

print("\n" + "="*80)
print("STEP 5: ENHANCED EDA WITH ACTIONABLE INSIGHTS")
print("="*80)

total_by_service = df_clean[service_cols].sum()
mean_by_service = df_clean[service_cols].mean()
std_by_service = df_clean[service_cols].std()

print("\n" + "-"*80)
print("TOTAL JOURNEYS BY SERVICE TYPE (Descending)")
print("-"*80)
total_sorted = total_by_service.sort_values(ascending=False)
for service, total in total_sorted.items():
    percentage = (total / total_by_service.sum()) * 100
    print(f"{service:20s}: {total:>15,.0f} ({percentage:5.2f}%)")

# Day of week analysis
print("\n" + "-"*80)
print("DAY OF WEEK ANALYSIS - AVERAGE PASSENGERS")
print("-"*80)
dow_analysis = df_clean.groupby('DayName')[service_cols].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

print(dow_analysis.round(0).to_string())

# Best and worst days for each service
print("\n" + "-"*80)
print("ACTIONABLE INSIGHTS: BEST DAYS TO TRAVEL (Less Crowded)")
print("-"*80)
for col in ['Local Route', 'Light Rail', 'Rapid Route']:
    avg_by_day = df_clean.groupby('DayName')[col].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    least_crowded = avg_by_day.idxmin()
    most_crowded = avg_by_day.idxmax()
    reduction = ((avg_by_day.max() - avg_by_day.min()) / avg_by_day.max()) * 100
    
    print(f"\n{col}:")
    print(f"  Least crowded day: {least_crowded} ({avg_by_day.min():.0f} avg passengers)")
    print(f"  Most crowded day: {most_crowded} ({avg_by_day.max():.0f} avg passengers)")
    print(f"  Reduction by choosing {least_crowded}: {reduction:.1f}% fewer passengers")

# Monthly patterns
print("\n" + "-"*80)
print("MONTHLY PATTERNS - AVERAGE PASSENGERS")
print("-"*80)
monthly_avg = df_clean.groupby('Month')[service_cols].mean()
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

for col in ['Local Route', 'Light Rail', 'Rapid Route']:
    col_total = monthly_avg[col].sum()
    peak_month = monthly_avg[col].idxmax()
    low_month = monthly_avg[col].idxmin()
    
    print(f"\n{col}:")
    print(f"  Peak month: {month_names[peak_month]} ({monthly_avg[col].loc[peak_month]:.0f} avg)")
    print(f"  Low month: {month_names[low_month]} ({monthly_avg[col].loc[low_month]:.0f} avg)")

# Weekend vs Weekday comparison
print("\n" + "-"*80)
print("WEEKEND VS WEEKDAY COMPARISON")
print("-"*80)
weekend_comparison = df_clean.groupby('IsWeekend')[service_cols].mean()
weekend_ratio = weekend_comparison.loc[0] / weekend_comparison.loc[1]

print("\nWeekday averages:")
print(weekend_comparison.loc[0].round(0).to_string())
print("\nWeekend averages:")
print(weekend_comparison.loc[1].round(0).to_string())
print("\nWeekday/Weekend ratio (how many times busier on weekdays):")
print(weekend_ratio.round(2).to_string())

# Peak hours insights (if we had time data, but we can infer from day patterns)
print("\n" + "-"*80)
print("ACTIONABLE INSIGHTS: TRAVEL RECOMMENDATIONS")
print("-"*80)
print("\n1. BEST DAYS TO AVOID CROWDS:")
for col in ['Local Route', 'Light Rail', 'Rapid Route']:
    avg_by_day = df_clean.groupby('DayName')[col].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    best_day = avg_by_day.idxmin()
    print(f"   {col}: Travel on {best_day} for {((avg_by_day.max() - avg_by_day.min()) / avg_by_day.max() * 100):.1f}% fewer passengers")

print("\n2. WEEKEND ADVANTAGE:")
for col in ['Local Route', 'Light Rail', 'Rapid Route']:
    weekday_avg = df_clean[df_clean['IsWeekend'] == 0][col].mean()
    weekend_avg = df_clean[df_clean['IsWeekend'] == 1][col].mean()
    reduction = ((weekday_avg - weekend_avg) / weekday_avg) * 100
    print(f"   {col}: Weekends have {reduction:.1f}% fewer passengers than weekdays")

# Visualizations for insights
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Actionable Insights Visualization', fontsize=16, fontweight='bold')

# Day of week comparison
ax1 = axes[0, 0]
dow_plot = df_clean.groupby('DayName')[['Local Route', 'Light Rail', 'Rapid Route']].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)
dow_plot.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_title('Average Passengers by Day of Week', fontweight='bold')
ax1.set_xlabel('Day of Week')
ax1.set_ylabel('Average Passengers')
ax1.legend(title='Service Type')
ax1.grid(True, alpha=0.3, axis='y')
ax1.tick_params(axis='x', rotation=45)

# Weekend vs Weekday
ax2 = axes[0, 1]
weekend_data = df_clean.groupby('IsWeekend')[['Local Route', 'Light Rail', 'Rapid Route']].mean()
weekend_data.index = ['Weekday', 'Weekend']
weekend_data.plot(kind='bar', ax=ax2, width=0.6)
ax2.set_title('Weekday vs Weekend Comparison', fontweight='bold')
ax2.set_xlabel('Day Type')
ax2.set_ylabel('Average Passengers')
ax2.legend(title='Service Type')
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', rotation=0)

# Monthly trends
ax3 = axes[1, 0]
monthly_plot = df_clean.groupby('Month')[['Local Route', 'Light Rail', 'Rapid Route']].mean()
monthly_plot.plot(kind='line', ax=ax3, marker='o', markersize=4)
ax3.set_title('Monthly Average Trends', fontweight='bold')
ax3.set_xlabel('Month')
ax3.set_ylabel('Average Passengers')
ax3.legend(title='Service Type')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels([month_names[i] for i in range(1, 13)], rotation=45)

# Service comparison
ax4 = axes[1, 1]
total_plot = total_by_service[['Local Route', 'Light Rail', 'Rapid Route', 'School', 'Peak Service']].sort_values(ascending=False)
total_plot.plot(kind='barh', ax=ax4, width=0.7)
ax4.set_title('Total Journeys by Service Type', fontweight='bold')
ax4.set_xlabel('Total Passengers')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('actionable_insights.png', dpi=300, bbox_inches='tight')
print("\n[OK] Saved: actionable_insights.png")

print("\n" + "="*80)
print("STEP 6: SARIMA FORECASTING")
print("="*80)
forecast_services = ['Local Route', 'Light Rail', 'Peak Service', 'Rapid Route', 'School']
if df_clean.shape[0] > 0:
    rev = df_clean[forecast_services].iloc[::-1]
    all_zero = (rev == 0).all(axis=1)
    zero_run = 0
    for flag in all_zero:
        if flag:
            zero_run += 1
        else:
            break
    if zero_run > 0:
        print(f"\nDetected {zero_run} trailing day(s) where all main services are zero. Dropping these rows before model training.")
        df_clean = df_clean.iloc[:-zero_run]
        df_clean.reset_index().rename(columns={'index': 'Date'}).to_csv('cleaned_daily_by_service.csv', index=False)
        print("[OK] cleaned_daily_by_service.csv updated after dropping trailing zeros")

def find_best_sarima_params(ts, seasonal_period=7, max_p=3, max_d=2, max_q=3, 
                             max_P=2, max_D=1, max_Q=2):
    """Find best SARIMA parameters using AIC"""
    best_aic = np.inf
    best_params = None
    
    # Test a subset of parameters (full grid search would take too long)
    p_values = range(0, max_p + 1)
    d_values = range(0, max_d + 1)
    q_values = range(0, max_q + 1)
    P_values = range(0, max_P + 1)
    D_values = range(0, max_D + 1)
    Q_values = range(0, max_Q + 1)
    
    param_combinations = list(itertools.product(p_values, d_values, q_values,
                                               P_values, D_values, Q_values))
    
    # Limit to reasonable combinations
    print(f"Testing {min(50, len(param_combinations))} parameter combinations...")
    
    tested = 0
    for params in param_combinations[:50]:
        p, d, q, P, D, Q = params
        try:
            model = SARIMAX(ts, order=(p, d, q), 
                          seasonal_order=(P, D, Q, seasonal_period),
                          enforce_stationarity=False, enforce_invertibility=False)
            fitted_model = model.fit(disp=False, maxiter=50)
            if fitted_model.aic < best_aic:
                best_aic = fitted_model.aic
                best_params = params
            tested += 1
        except:
            continue
    
    if best_params is None:
        return (1, 1, 1, 1, 1, 1)
    
    print(f"Best AIC: {best_aic:.2f}, Parameters: {best_params}")
    return best_params

def forecast_sarima(series, forecast_periods=7, seasonal_period=7):
        series = series.dropna()
    
    if len(series) < 50:
        print(f"  Warning: Series too short ({len(series)}), using simple average")
        last_value = series.iloc[-1]
        return pd.Series([last_value] * forecast_periods)
    
    adf_result = adfuller(series.dropna())
    is_stationary = adf_result[1] < 0.05
    
    try:
        orders_to_try = [
            ((1, 1, 1), (1, 1, 1, seasonal_period)),
            ((2, 1, 2), (1, 1, 1, seasonal_period)),
            ((1, 0, 1), (1, 0, 1, seasonal_period)),
            ((0, 1, 1), (0, 1, 1, seasonal_period)),
        ]
        
        best_model = None
        best_aic = np.inf
        
        for order, seasonal_order in orders_to_try:
            try:
                model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                              enforce_stationarity=False, enforce_invertibility=False)
                fitted = model.fit(disp=False, maxiter=100)
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
            except:
                continue
        
        if best_model is None:
            # Fallback
            model = SARIMAX(series, order=(1, 1, 1), 
                          seasonal_order=(1, 1, 1, seasonal_period),
                          enforce_stationarity=False, enforce_invertibility=False)
            best_model = model.fit(disp=False, maxiter=100)
        # Forecast
        forecast = best_model.forecast(steps=forecast_periods)
        forecast_ci = best_model.get_forecast(steps=forecast_periods).conf_int()

        return forecast, forecast_ci, best_model

    except Exception as e:
        print(f"  Error in SARIMA fitting: {e}")
        # Fallback: simple moving average
        last_values = series.tail(seasonal_period).mean()
        return pd.Series([last_values] * forecast_periods), None, None

# Train-test split for validation
split_date = df_clean.index.max() - timedelta(days=30)
train_data = df_clean[df_clean.index <= split_date]
test_data = df_clean[df_clean.index > split_date]

print(f"\nTraining period: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} days)")
print(f"Test period: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} days)")

# Forecast for each service
forecast_services = ['Local Route', 'Light Rail', 'Peak Service', 'Rapid Route', 'School']
forecast_results = {}
sarima_models = {}

for col in forecast_services:
    print(f"\n{'='*60}")
    print(f"FORECASTING: {col}")
    print('='*60)
    
    # Training series
    train_series = train_data[col].dropna()
    
    if len(train_series) < 50:
        print(f"  Series too short, skipping...")
        continue
    
    # Fit SARIMA model
    print(f"  Fitting SARIMA model...")
    try:
        forecast, forecast_ci, model = forecast_sarima(train_series, forecast_periods=7, seasonal_period=7)
        
        if model is not None:
            # Validate on test set
            test_series = test_data[col].dropna()
            if len(test_series) > 0:
                test_forecast = model.forecast(steps=min(7, len(test_series)))
                mae = mean_absolute_error(test_series.iloc[:len(test_forecast)], test_forecast)
                rmse = np.sqrt(mean_squared_error(test_series.iloc[:len(test_forecast)], test_forecast))
                
                print(f"  Validation MAE: {mae:.2f}")
                print(f"  Validation RMSE: {rmse:.2f}")
            
            # Full forecast from end of data
            full_series = df_clean[col].dropna()
            final_forecast, final_ci, final_model = forecast_sarima(full_series, forecast_periods=7, seasonal_period=7)

            # Create forecast dates
            last_date = df_clean.index.max()
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=7, freq='D')

            # Sanitize forecasts: if SARIMA returns non-positive or NaN values, replace per-day using
            # recent historical averages for the same weekday (fallback)
            sanitized = final_forecast.copy()
            for i in range(len(sanitized)):
                v = sanitized.iloc[i]
                if pd.isna(v) or v <= 0:
                    target_date = forecast_dates[i]
                    weekday = target_date.dayofweek
                    # take last 8 occurrences of same weekday
                    same_weekday = full_series[full_series.index.dayofweek == weekday]
                    if len(same_weekday) >= 4:
                        replacement = same_weekday.tail(8).mean()
                    elif len(same_weekday) > 0:
                        replacement = same_weekday.mean()
                    else:
                        replacement = full_series.mean()
                    # if still NaN, fallback to median
                    if pd.isna(replacement) or replacement <= 0:
                        replacement = full_series.median() if not pd.isna(full_series.median()) else 0
                    sanitized.iloc[i] = replacement

            # Ensure non-negative and as numpy array
            sanitized_vals = np.maximum(sanitized.values, 0)
            lower_vals = None
            upper_vals = None
            if final_ci is not None:
                # clip CIs to be non-negative
                lower_vals = np.maximum(final_ci.iloc[:, 0].values, 0)
                upper_vals = np.maximum(final_ci.iloc[:, 1].values, 0)

            forecast_results[col] = {
                'forecast': sanitized_vals,
                'dates': forecast_dates,
                'lower_ci': lower_vals,
                'upper_ci': upper_vals,
                'model': final_model
            }
            
            sarima_models[col] = final_model
            
            print(f"  Forecast mean: {final_forecast.mean():.2f}")
            print(f"  Forecast range: {final_forecast.min():.2f} - {final_forecast.max():.2f}")
            
    except Exception as e:
        print(f"  Error: {e}")
        # Fallback forecast
        last_value = train_series.iloc[-1]
        forecast_dates = pd.date_range(start=df_clean.index.max() + timedelta(days=1), periods=7, freq='D')
        forecast_results[col] = {
            'forecast': [last_value] * 7,
            'dates': forecast_dates,
            'lower_ci': None,
            'upper_ci': None,
            'model': None
        }

print("\n" + "="*80)
print("STEP 7: RESIDUAL DIAGNOSTICS")
print("="*80)

for col in forecast_services:
    if col in sarima_models and sarima_models[col] is not None:
        print(f"\n{col} - Residual Diagnostics:")
        model = sarima_models[col]
        residuals = model.resid
        
        try:
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
            p_value = lb_test['lb_pvalue'].iloc[-1]
            print(f"  Ljung-Box test p-value: {p_value:.4f}")
            if p_value > 0.05:
                print(f"    [OK] Residuals are white noise (good fit)")
            else:
                print(f"    [WARNING] Residuals show autocorrelation (model may need improvement)")
        except:
            pass
        
        print(f"  Residual mean: {residuals.mean():.4f}")
        print(f"  Residual std: {residuals.std():.4f}")

print("\n" + "="*80)
print("STEP 8: GENERATING FORECAST OUTPUT")
print("="*80)
all_rows = []
for col in forecast_services:
    if col in forecast_results:
        dates = forecast_results[col]['dates']
        forecasts = np.round(forecast_results[col]['forecast']).astype(int)
        lower = forecast_results[col]['lower_ci']
        upper = forecast_results[col]['upper_ci']

        for i, d in enumerate(dates):
            row = {
                'Service': col,
                'Date': d.strftime('%Y-%m-%d'),
                'Forecast': int(max(0, forecasts[i]))
            }
            if lower is not None:
                try:
                    l = int(round(lower[i]))
                    u = int(round(upper[i]))
                    # ensure ordering
                    l, u = (min(l, u), max(l, u))
                    row['Lower_CI'] = int(max(0, l))
                    row['Upper_CI'] = int(max(0, u))
                except Exception:
                    row['Lower_CI'] = ''
                    row['Upper_CI'] = ''
            else:
                row['Lower_CI'] = ''
                row['Upper_CI'] = ''
            all_rows.append(row)

# Save only combined per-service forecast as single CSV
forecast_all_df = pd.DataFrame(all_rows)
forecast_all_df.to_csv('forecast_7days_all.csv', index=False)
print(f"[OK] Saved combined forecasts to: forecast_7days_all.csv")

# Print forecast table (wide format) for step 8
try:
    forecast_wide = forecast_all_df.pivot(index='Date', columns='Service', values='Forecast')
    print("\n" + "="*40)
    print("7-DAY FORECAST (combined)")
    print("="*40)
    # ensure dates sorted
    forecast_wide = forecast_wide.reindex(sorted(forecast_wide.index))
    print(forecast_wide.to_string())
except Exception as e:
    print(f"Could not print wide forecast table: {e}")

# Also save a wide-format CSV with Lower/Upper columns per service (one row per date)
try:
    dates = sorted(forecast_all_df['Date'].unique())
    wide_df = pd.DataFrame({'Date': dates})
    for svc in forecast_services:
        svc_df = forecast_all_df[forecast_all_df['Service'] == svc].set_index('Date')
        # reindex to ensure all dates present in order
        svc_df = svc_df.reindex(dates)
        wide_df[svc] = pd.to_numeric(svc_df['Forecast'], errors='coerce').fillna(0).astype(int).values
        # Lower/Upper may be empty strings
        wide_df[f'{svc}_Lower'] = pd.to_numeric(svc_df['Lower_CI'], errors='coerce').fillna(0).astype(int).values
        wide_df[f'{svc}_Upper'] = pd.to_numeric(svc_df['Upper_CI'], errors='coerce').fillna(0).astype(int).values

    wide_df.to_csv('7_day_forecast_sarima.csv', index=False)
    print('[OK] Also saved wide-format forecast to: 7_day_forecast_sarima.csv')
except Exception as e:
    print(f'Could not save wide-format forecast CSV: {e}')

# ============================================================================
# 9. FORECAST VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("STEP 9: FORECAST VISUALIZATION")
print("="*80)

# Plot only the 7-day forecast lines (one line per service)
try:
    forecast_all_df['Date_dt'] = pd.to_datetime(forecast_all_df['Date'])
    pivot = forecast_all_df.pivot(index='Date_dt', columns='Service', values='Forecast')
    fig, ax = plt.subplots(figsize=(10, 6))
    for svc in pivot.columns:
        ax.plot(pivot.index, pivot[svc], marker='o', label=svc)
    ax.set_title('7-day Forecast - All Services')
    ax.set_xlabel('Date')
    ax.set_ylabel('Forecast passengers')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig('7_day_forecast_lines.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Saved: 7_day_forecast_lines.png")
except Exception as e:
    print(f"Could not create forecast line plot: {e}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*80)
print("\nGenerated Files:")
print("  1. cleaned_daily_by_service.csv - Cleaned continuous daily time series")
print("  2. actionable_insights.png - Key insights visualization")
print("  3. forecast_7days_all.csv - Combined per-service 7-day forecast (Service, Date, Forecast, Lower_CI, Upper_CI)")
print("  4. 7_day_forecast_lines.png - 7-day forecast line chart (all services)")
print("  5. Technical_Report_SARIMA.txt - Technical documentation (generate via generate_report.py)")
print("\n" + "="*80)
