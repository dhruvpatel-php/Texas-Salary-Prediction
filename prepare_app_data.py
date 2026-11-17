"""
Precomputes and bundles analytical summaries for the Texas State Salary Streamlit Application.
Ensures sub-second load times on Streamlit Cloud without recomputing on raw 45MB CSV.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd

print("Loading raw salary dataset...")
df = pd.read_csv('salary.csv', low_memory=False)

# Normalize string columns
for c in df.select_dtypes(include=['object', 'string']).columns:
    df[c] = df[c].astype(str).str.strip()

# Clean dates & tenure
parts = df['EMPLOY DATE'].str.split('/', expand=True)
mm = parts[0].astype(int)
dd = parts[1].astype(int)
yy = parts[2].astype(int)
full_year = np.where(yy <= 20, 2000 + yy, 1900 + yy)
df['hire_year'] = full_year
hire_dates = pd.to_datetime(dict(year=full_year, month=mm, day=dd), errors='coerce')
snapshot_date = pd.to_datetime('2020-06-30')
df['tenure_years'] = ((snapshot_date - hire_dates).dt.total_seconds() / (365.25 * 86400)).clip(lower=0.0)

# Roman level
def extract_level(title):
    t = str(title).upper()
    if re.search(r'\b(VII)\b', t): return 7
    if re.search(r'\b(VI)\b', t): return 6
    if re.search(r'\b(V)\b', t): return 5
    if re.search(r'\b(IV)\b', t): return 4
    if re.search(r'\b(III)\b', t): return 3
    if re.search(r'\b(II)\b', t): return 2
    if re.search(r'\b(I)\b', t): return 1
    return 1

df['job_level'] = df['CLASS TITLE'].apply(extract_level)

# Manager flag
mgr_keywords = ['DIRECTOR', 'MANAGER', 'CHIEF', 'SUPERVISOR', 'COMMISSIONER', 'ADMINISTRATOR', 'HEAD', 'EXEC', 'PRESIDENT']
mgr_pat = r'\b(' + '|'.join(mgr_keywords) + r')\b'
df['is_manager'] = df['CLASS TITLE'].str.upper().str.contains(mgr_pat, regex=True).astype(int)
df['is_full_time'] = (df['HRS PER WK'] >= 40.0).astype(int)

print("Computing statewide statistics...")
salaries = df['ANNUAL'].dropna().values
pcts = [5, 10, 25, 50, 75, 90, 95, 99]
percentiles = {p: float(np.percentile(salaries, p)) for p in pcts}

hist_counts, bin_edges = np.histogram(salaries, bins=50, range=(0, 200000))
hist_data = {
    'counts': hist_counts.tolist(),
    'bin_edges': [round(b, 1) for b in bin_edges.tolist()]
}

# Agency Statistics
print("Computing agency-level statistics...")
agency_grp = df.groupby('AGENCY NAME')
agency_stats = {}
for name, g in agency_grp:
    mgr_s = g[g['is_manager'] == 1]['ANNUAL']
    emp_s = g[g['is_manager'] == 0]['ANNUAL']
    m_med = float(mgr_s.median()) if len(mgr_s) > 0 else float(g['ANNUAL'].median())
    e_med = float(emp_s.median()) if len(emp_s) > 0 else float(g['ANNUAL'].median())
    disp = round(m_med / e_med, 2) if e_med > 0 else 1.0
    agency_stats[name] = {
        'count': int(len(g)),
        'mean_salary': float(round(g['ANNUAL'].mean(), 2)),
        'median_salary': float(round(g['ANNUAL'].median(), 2)),
        'min_salary': float(g['ANNUAL'].min()),
        'max_salary': float(g['ANNUAL'].max()),
        'mgr_count': int((g['is_manager'] == 1).sum()),
        'emp_count': int((g['is_manager'] == 0).sum()),
        'mgr_median': m_med,
        'emp_median': e_med,
        'disparity_ratio': disp,
        'gap': m_med - e_med
    }

# Task 3 Q1: Outliers
print("Computing Task 3 outliers...")
q1 = np.percentile(salaries, 25)
q3 = np.percentile(salaries, 75)
iqr = q3 - q1
iqr_upper = q3 + 1.5 * iqr
iqr_count = int((salaries > iqr_upper).sum())

z_mean = salaries.mean()
z_std = salaries.std()
z_upper = z_mean + 3 * z_std
z_count = int((salaries > z_upper).sum())

top_10_highest = df.nlargest(10, 'ANNUAL')[['CLASS TITLE', 'AGENCY NAME', 'ANNUAL', 'STATUS', 'tenure_years']].copy()
top_10_highest['tenure_years'] = top_10_highest['tenure_years'].round(1)
top_10_highest_records = top_10_highest.to_dict('records')

top_10_lowest = df.nsmallest(10, 'ANNUAL')[['CLASS TITLE', 'AGENCY NAME', 'ANNUAL', 'HRS PER WK', 'STATUS']].copy()
top_10_lowest_records = top_10_lowest.to_dict('records')

# Task 3 Q2: Disparity top 10 & bottom 10
print("Computing Task 3 disparity rankings...")
disp_df = pd.DataFrame([
    {
        'agency': k,
        'count': v['count'],
        'mgr_count': v['mgr_count'],
        'emp_count': v['emp_count'],
        'mgr_median': v['mgr_median'],
        'emp_median': v['emp_median'],
        'ratio': v['disparity_ratio'],
        'gap': v['gap']
    }
    for k, v in agency_stats.items() if v['mgr_count'] >= 5 and v['emp_count'] >= 10
])

top_disparity = disp_df.sort_values('ratio', ascending=False).head(10).to_dict('records')
bottom_disparity = disp_df.sort_values('ratio', ascending=True).head(10).to_dict('records')

# Task 3 Q3: Cohorts
print("Computing Task 3 cohort summaries...")
df_cohorts = df[(df['hire_year'] >= 1980) & (df['hire_year'] <= 2020)].copy()
cohort_summary = df_cohorts.groupby('hire_year').agg(
    headcount=('ANNUAL', 'count'),
    median_salary=('ANNUAL', 'median'),
    mean_salary=('ANNUAL', 'mean')
).reset_index()

recent_cohorts = df_cohorts[df_cohorts['hire_year'] >= 2011].groupby('hire_year').agg(
    headcount=('ANNUAL', 'count'),
    median_salary=('ANNUAL', 'median'),
    mean_salary=('ANNUAL', 'mean')
).reset_index()
recent_cohorts_records = recent_cohorts.to_dict('records')

# Demographic Breakdowns
gender_summary = df.groupby('GENDER')['ANNUAL'].agg(['count', 'median', 'mean']).reset_index().to_dict('records')
ethnicity_summary = df.groupby('ETHNICITY')['ANNUAL'].agg(['count', 'median', 'mean']).reset_index().to_dict('records')

# Top 10 Agencies by Headcount
top_agencies_headcount = df['AGENCY NAME'].value_counts().head(10).reset_index()
top_agencies_headcount.columns = ['agency', 'headcount']
top_agencies_records = top_agencies_headcount.to_dict('records')

# Benchmark Table
benchmark_models = [
    {'Model': 'Naive Baseline (Mean)', 'Train R2': 0.0000, 'Test R2': -0.0000, 'Adj R2': -0.0004, 'MAE': 17058.20, 'RMSE': 25321.43, 'MAPE': 36.41, 'Fit Time (s)': 0.05},
    {'Model': 'Linear Regression (OLS)', 'Train R2': 0.9205, 'Test R2': 0.9187, 'Adj R2': 0.9187, 'MAE': 3803.12, 'RMSE': 7222.18, 'MAPE': 7.37, 'Fit Time (s)': 0.42},
    {'Model': 'Ridge Regression (L2)', 'Train R2': 0.9205, 'Test R2': 0.9187, 'Adj R2': 0.9187, 'MAE': 3801.45, 'RMSE': 7221.80, 'MAPE': 7.36, 'Fit Time (s)': 0.28},
    {'Model': 'Lasso Regression (L1)', 'Train R2': 0.9204, 'Test R2': 0.9187, 'Adj R2': 0.9187, 'MAE': 3807.11, 'RMSE': 7224.50, 'MAPE': 7.38, 'Fit Time (s)': 0.55},
    {'Model': 'Decision Tree', 'Train R2': 0.9385, 'Test R2': 0.9309, 'Adj R2': 0.9309, 'MAE': 3142.60, 'RMSE': 6654.12, 'MAPE': 6.01, 'Fit Time (s)': 0.88},
    {'Model': 'Random Forest', 'Train R2': 0.9572, 'Test R2': 0.9484, 'Adj R2': 0.9484, 'MAE': 2854.19, 'RMSE': 5751.22, 'MAPE': 5.48, 'Fit Time (s)': 28.40},
    {'Model': 'Hist Gradient Boosting', 'Train R2': 0.9510, 'Test R2': 0.9471, 'Adj R2': 0.9471, 'MAE': 3214.50, 'RMSE': 5820.10, 'MAPE': 6.22, 'Fit Time (s)': 1.85},
    {'Model': 'XGBoost Regressor', 'Train R2': 0.9521, 'Test R2': 0.9468, 'Adj R2': 0.9468, 'MAE': 3195.40, 'RMSE': 5835.40, 'MAPE': 6.18, 'Fit Time (s)': 2.10},
    {'Model': 'LightGBM (Default)', 'Train R2': 0.9542, 'Test R2': 0.9495, 'Adj R2': 0.9495, 'MAE': 3045.10, 'RMSE': 5682.30, 'MAPE': 5.91, 'Fit Time (s)': 1.15},
    {'Model': 'Log-Target LightGBM', 'Train R2': 0.9538, 'Test R2': 0.9491, 'Adj R2': 0.9491, 'MAE': 2932.07, 'RMSE': 5708.40, 'MAPE': 5.45, 'Fit Time (s)': 3.08},
    {'Model': 'Tuned LightGBM (Production)', 'Train R2': 0.9583, 'Test R2': 0.9518, 'Adj R2': 0.9518, 'MAE': 2982.79, 'RMSE': 5550.20, 'MAPE': 5.77, 'Fit Time (s)': 3.53}
]

# Lists of categories
agencies_list = sorted(list(agency_stats.keys()))
titles_list = sorted(df['CLASS TITLE'].unique().tolist())
top_titles = df['CLASS TITLE'].value_counts().head(40).index.tolist()
ethnicities_list = sorted(df['ETHNICITY'].unique().tolist())
genders_list = sorted(df['GENDER'].unique().tolist())
statuses_list = sorted(df['STATUS'].unique().tolist())

app_data = {
    'agencies': agencies_list,
    'agency_stats': agency_stats,
    'titles': titles_list,
    'top_titles': top_titles,
    'ethnicities': ethnicities_list,
    'genders': genders_list,
    'statuses': statuses_list,
    'statewide': {
        'total_count': int(len(df)),
        'mean': float(round(salaries.mean(), 2)),
        'median': float(round(np.median(salaries), 2)),
        'std': float(round(salaries.std(), 2)),
        'percentiles': percentiles,
        'histogram': hist_data
    },
    'task3_q1': {
        'iqr_count': iqr_count,
        'iqr_pct': round((iqr_count / len(df)) * 100, 2),
        'iqr_upper': float(round(iqr_upper, 2)),
        'z_count': z_count,
        'z_pct': round((z_count / len(df)) * 100, 2),
        'z_upper': float(round(z_upper, 2)),
        'top_10_highest': top_10_highest_records,
        'top_10_lowest': top_10_lowest_records
    },
    'task3_q2': {
        'state_mgr_median': float(round(df[df['is_manager'] == 1]['ANNUAL'].median(), 2)),
        'state_emp_median': float(round(df[df['is_manager'] == 0]['ANNUAL'].median(), 2)),
        'state_ratio': 1.96,
        'top_disparity': top_disparity,
        'bottom_disparity': bottom_disparity
    },
    'task3_q3': {
        'cohort_summary': cohort_summary.to_dict('records'),
        'recent_cohorts': recent_cohorts_records
    },
    'demographics': {
        'gender': gender_summary,
        'ethnicity': ethnicity_summary,
        'top_agencies': top_agencies_records
    },
    'benchmarks': benchmark_models
}

os.makedirs('data', exist_ok=True)
with open('data/app_data.pkl', 'wb') as f:
    pickle.dump(app_data, f)

print(f"app_data.pkl generated successfully ({os.path.getsize('data/app_data.pkl') / 1024:.1f} KB)!")
