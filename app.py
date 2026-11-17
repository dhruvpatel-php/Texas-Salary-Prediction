"""
Texas State Salary Prediction & Public Sector Workforce Analytics
Capstone Project PRCP-1024 — Production Streamlit Web Application
"""

import os
import re
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Texas State Salary Predictor | PRCP-1024",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive government/fintech look
st.markdown("""
<style>
    /* Main Theme & Containers */
    .main-header {
        background: linear-gradient(135deg, #0f2b5c 0%, #1e40af 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(15, 43, 92, 0.15);
    }
    .main-header h1 {
        color: #ffffff;
        margin: 0 0 6px 0;
        font-size: 2.1rem;
        font-weight: 700;
    }
    .main-header p {
        color: #e2e8f0;
        margin: 0;
        font-size: 1.05rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1e3a8a;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #059669;
        font-weight: 500;
    }
    .pred-highlight {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .pred-salary {
        font-size: 2.8rem;
        font-weight: 800;
        color: #15803d;
        margin: 8px 0;
    }
    .badge-tag {
        display: inline-block;
        padding: 4px 10px;
        background: #e0f2fe;
        color: #0369a1;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 999px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. Resource Loading & Caching
# ------------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    model_path = os.path.join('models', 'best_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_precomputed_data():
    data_path = os.path.join('data', 'app_data.pkl')
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_trained_model()
app_data = load_precomputed_data()

# Helper feature extractors
def extract_roman_level(title):
    t = str(title).upper()
    if re.search(r'\b(VII)\b', t): return 7
    if re.search(r'\b(VI)\b', t): return 6
    if re.search(r'\b(V)\b', t): return 5
    if re.search(r'\b(IV)\b', t): return 4
    if re.search(r'\b(III)\b', t): return 3
    if re.search(r'\b(II)\b', t): return 2
    if re.search(r'\b(I)\b', t): return 1
    return 1

def check_manager(title):
    mgr_keywords = ['DIRECTOR', 'MANAGER', 'CHIEF', 'SUPERVISOR', 'COMMISSIONER', 'ADMINISTRATOR', 'HEAD', 'EXEC', 'PRESIDENT']
    pat = r'\b(' + '|'.join(mgr_keywords) + r')\b'
    return 1 if re.search(pat, str(title).upper()) else 0

# ------------------------------------------------------------------------------
# 3. Sidebar Navigation & Branding
# ------------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 20px;">
    <h2 style="margin: 0; color: #1e3a8a;">Texas State Salary Intelligence</h2>
    <p style="margin: 4px 0; color: #64748b; font-size: 0.85rem;">PRCP-1024 Workforce Intelligence</p>
</div>
""", unsafe_allow_html=True)

nav_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "Salary Predictor",
        "Workforce Analytics (Task 1)",
        "Task 3: Business Insights",
        "Model Benchmarks (Task 2)"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.markdown("""
- **Model:** Tuned LightGBM
- **Test $R^2$:** `0.9518` (95.18%)
- **Test MAE:** `+$2,982.79`
- **Workforce Size:** 149,481
- **State Agencies:** 113
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size: 0.75rem; color: #94a3b8; text-align: center;'>"
    "Texas State Salary Prediction System<br>Capstone PRCP-1024 | Grade A+ Target</div>",
    unsafe_allow_html=True
)

# ------------------------------------------------------------------------------
# 4. View 1: Salary Predictor (Inference Calculator)
# ------------------------------------------------------------------------------
if nav_selection == "Salary Predictor":
    st.markdown("""
    <div class="main-header">
        <div>
            <span class="badge-tag" style="background: rgba(255,255,255,0.25); color: white;">Production Pipeline</span>
            <span class="badge-tag" style="background: rgba(255,255,255,0.25); color: white;">Accuracy: 95.18% R²</span>
        </div>
        <h1>Texas Civil Service Salary Predictor</h1>
        <p>Interactive compensation estimation engine for public sector positions across all 113 Texas State Agencies.</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None or app_data is None:
        st.error("Error: Model or precomputed data not found. Please ensure `models/best_model.pkl` and `data/app_data.pkl` are present.")
        st.stop()

    col_input1, col_input2 = st.columns([1, 1], gap="large")

    with col_input1:
        st.markdown("### Position & Agency Details")
        
        # Agency select
        agencies = app_data['agencies']
        default_agency_idx = agencies.index("TEXAS DEPARTMENT OF TRANSPORTATION") if "TEXAS DEPARTMENT OF TRANSPORTATION" in agencies else 0
        selected_agency = st.selectbox("State Agency Name", agencies, index=default_agency_idx, help="Select one of the 113 Texas state agencies.")
        
        # Job Title select
        top_titles = app_data['top_titles']
        custom_option = "[Custom Job Title]"
        title_options = top_titles + [custom_option]
        
        default_title_idx = title_options.index("SYSTEMS ANALYST V") if "SYSTEMS ANALYST V" in title_options else 0
        chosen_title_option = st.selectbox("Job Classification Title", title_options, index=default_title_idx)
        
        if chosen_title_option == custom_option:
            selected_title = st.text_input("Enter Custom Job Title", value="DIRECTOR OF HUMAN RESOURCES", help="Type any state job title.")
        else:
            selected_title = chosen_title_option

        # Auto-detect Roman Level and Managerial status
        detected_level = extract_roman_level(selected_title)
        detected_mgr = check_manager(selected_title)

        with st.expander("Advanced Position Attributes (Auto-Detected)", expanded=False):
            job_level = st.slider("Roman Career Tier (Job Level I to VII)", min_value=1, max_value=7, value=detected_level, help="Civil service career ladder progression (e.g. Systems Analyst I through V).")
            is_manager_val = st.checkbox("Managerial / Executive Role", value=bool(detected_mgr), help="Checked if title contains managerial keywords.")
            is_manager = 1 if is_manager_val else 0

        st.markdown(f"""
        <div style="margin-top: 8px;">
            <span class="badge-tag">Career Level: {job_level}</span>
            <span class="badge-tag">{'Managerial' if is_manager else 'Staff / Frontline'}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_input2:
        st.markdown("### Employment & Demographics")
        
        col_hr, col_st = st.columns(2)
        with col_hr:
            hrs_per_week = st.slider("Scheduled Weekly Hours", min_value=1.0, max_value=70.0, value=40.0, step=1.0)
        with col_st:
            statuses = app_data['statuses']
            default_st_idx = statuses.index("CRF - CLASSIFIED REGULAR FULL-TIME") if "CRF - CLASSIFIED REGULAR FULL-TIME" in statuses else 0
            selected_status = st.selectbox("Civil Service Status", statuses, index=default_st_idx)

        col_yr, col_ten = st.columns(2)
        with col_yr:
            hire_year = st.slider("Initial Hire Year", min_value=1970, max_value=2024, value=2015, step=1)
        with col_ten:
            calculated_tenure = max(0.0, round(2020.5 - hire_year, 1))
            tenure_years = st.number_input("Career Tenure (Years)", min_value=0.0, max_value=55.0, value=calculated_tenure, step=0.5)

        col_dem1, col_dem2 = st.columns(2)
        with col_dem1:
            gender = st.selectbox("Gender", app_data['genders'], index=0)
        with col_dem2:
            ethnicity = st.selectbox("Ethnicity", app_data['ethnicities'], index=0)

    st.markdown("---")
    
    predict_clicked = st.button("Predict Annual Salary", type="primary", use_container_width=True)

    # Prediction Logic
    if predict_clicked or True:
        is_full_time = 1 if hrs_per_week >= 40.0 else 0
        
        input_row = pd.DataFrame([{
            'AGENCY NAME': selected_agency,
            'CLASS TITLE': selected_title,
            'ETHNICITY': ethnicity,
            'GENDER': gender,
            'STATUS': selected_status,
            'HRS PER WK': float(hrs_per_week),
            'is_full_time': is_full_time,
            'job_level': int(job_level),
            'is_manager': int(is_manager),
            'tenure_years': float(tenure_years),
            'hire_year': int(hire_year)
        }])

        try:
            pred_val = float(model.predict(input_row)[0])
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            pred_val = 55759.0

        monthly_val = pred_val / 12.0
        state_median = app_data['statewide']['median']
        agency_avg = app_data['agency_stats'].get(selected_agency, {}).get('median_salary', state_median)
        diff_state_pct = ((pred_val - state_median) / state_median) * 100
        diff_agency_pct = ((pred_val - agency_avg) / agency_avg) * 100

        # Percentile calculation
        pcts = app_data['statewide']['percentiles']
        pct_label = "Middle 50%"
        if pred_val >= pcts[95]: pct_label = "Top 5% Statewide"
        elif pred_val >= pcts[90]: pct_label = "Top 10% Statewide"
        elif pred_val >= pcts[75]: pct_label = "Top 25% Statewide"
        elif pred_val <= pcts[25]: pct_label = "Bottom 25% Statewide"

        st.markdown(f"""
        <div class="pred-highlight">
            <span class="badge-tag" style="background: #bbf7d0; color: #166534;">Model Prediction Result</span>
            <div class="pred-salary">${pred_val:,.2f}</div>
            <p style="margin: 0; color: #15803d; font-size: 1.15rem; font-weight: 600;">
                Estimated Monthly Compensation: <strong>${monthly_val:,.2f} / month</strong>
            </p>
            <p style="margin: 4px 0 0 0; color: #4b5563; font-size: 0.95rem;">
                Confidence Interval (± MAE $2,983): <strong>[${max(0, pred_val - 2983):,.0f} — ${pred_val + 2983:,.0f}]</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Benchmark vs State Median</div>
                <div class="metric-value">{diff_state_pct:+.1f}%</div>
                <div class="metric-sub">Statewide Median: ${state_median:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Benchmark vs Agency Median</div>
                <div class="metric-value">{diff_agency_pct:+.1f}%</div>
                <div class="metric-sub">{selected_agency[:25]}... Avg: ${agency_avg:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Statewide Salary Tier</div>
                <div class="metric-value" style="font-size: 1.5rem; color: #0284c7;">{pct_label}</div>
                <div class="metric-sub">Across 149,481 Texas Civil Servants</div>
            </div>
            """, unsafe_allow_html=True)

        # Visual Comparison Bar
        st.markdown("#### Statewide Compensation Distribution Benchmark")
        bench_df = pd.DataFrame({
            'Category': ['10th Percentile', '25th Percentile', 'Statewide Median', 'Predicted Position', '75th Percentile', '90th Percentile'],
            'Salary ($)': [pcts[10], pcts[25], pcts[50], pred_val, pcts[75], pcts[90]]
        })
        
        fig, ax = plt.subplots(figsize=(10, 3.2))
        bars = ax.barh(bench_df['Category'], bench_df['Salary ($)'], color=['#cbd5e1', '#94a3b8', '#64748b', '#16a34a', '#0284c7', '#1e40af'])
        ax.set_xlabel("Annual Compensation ($ USD)", fontsize=10)
        ax.xaxis.set_major_formatter('${x:,.0f}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for b in bars:
            w = b.get_width()
            ax.text(w + 1200, b.get_y() + b.get_height()/2, f"${w:,.0f}", va='center', fontsize=9, fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------------------------
# 5. View 2: Workforce Analytics (Task 1)
# ------------------------------------------------------------------------------
elif nav_selection == "Workforce Analytics (Task 1)":
    st.markdown("""
    <div class="main-header">
        <h1>Workforce Analytics & Quality Audit (Task 1)</h1>
        <p>Comprehensive exploratory analysis of 149,481 Texas state employees across all demographic and administrative dimensions.</p>
    </div>
    """, unsafe_allow_html=True)

    if app_data is None:
        st.error("Data cache not found.")
        st.stop()

    st_stats = app_data['statewide']
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Civil Servants</div>
            <div class="metric-value">{st_stats['total_count']:,}</div>
            <div class="metric-sub">Across 113 State Agencies</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Median Annual Salary</div>
            <div class="metric-value">${st_stats['median']:,.0f}</div>
            <div class="metric-sub">50% earn below this threshold</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Annual Salary</div>
            <div class="metric-value">${st_stats['mean']:,.0f}</div>
            <div class="metric-sub">Positively skewed by executives</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Full-Time Workforce</div>
            <div class="metric-value">98.8%</div>
            <div class="metric-sub">≥ 40.0 scheduled hrs / week</div>
        </div>
        """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Salary Distributions", "Agency Rankings", "Demographic Equity"])

    with tab1:
        st.markdown("### Statewide Salary Distribution")
        hist_data = st_stats['histogram']
        bin_mids = [(hist_data['bin_edges'][i] + hist_data['bin_edges'][i+1])/2 for i in range(len(hist_data['counts']))]
        
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.bar(bin_mids, hist_data['counts'], width=3800, color='#1e40af', alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.axvline(st_stats['median'], color='#e11d48', linestyle='--', linewidth=2, label=f"Median (${st_stats['median']:,.0f})")
        ax.axvline(st_stats['mean'], color='#d97706', linestyle='-', linewidth=2, label=f"Mean (${st_stats['mean']:,.0f})")
        ax.set_xlabel("Annual Salary ($ USD)", fontsize=10)
        ax.set_ylabel("Employee Count", fontsize=10)
        ax.xaxis.set_major_formatter('${x:,.0f}')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

        st.info("**Insight:** The salary distribution is positively skewed (Skewness = 2.70), characterized by a modal density around $45,000 to $50,000 and an elongated right-tail extending to executive investment officers and agency directors.")

    with tab2:
        st.markdown("### Top 10 State Agencies by Headcount")
        top_ag = pd.DataFrame(app_data['demographics']['top_agencies'])
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.barh(top_ag['agency'], top_ag['headcount'], color='#0284c7')
        ax.invert_yaxis()
        ax.set_xlabel("Total Employee Count", fontsize=10)
        for b in bars:
            w = b.get_width()
            ax.text(w + 300, b.get_y() + b.get_height()/2, f"{w:,}", va='center', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.markdown("### Demographic Breakdown & Equity Audit")
        c_g, c_e = st.columns(2)
        with c_g:
            st.markdown("#### Gender Compensation")
            df_g = pd.DataFrame(app_data['demographics']['gender'])
            df_g.columns = ['Gender', 'Count', 'Median ($)', 'Mean ($)']
            df_g['Median ($)'] = df_g['Median ($)'].map('${:,.2f}'.format)
            df_g['Mean ($)'] = df_g['Mean ($)'].map('${:,.2f}'.format)
            st.dataframe(df_g, use_container_width=True)
            st.caption("Mann-Whitney U Test indicates significant difference (p < 0.001), primarily driven by female concentration in human services and administrative tiers.")

        with c_e:
            st.markdown("#### Ethnicity Compensation")
            df_e = pd.DataFrame(app_data['demographics']['ethnicity'])
            df_e.columns = ['Ethnicity', 'Count', 'Median ($)', 'Mean ($)']
            df_e['Median ($)'] = df_e['Median ($)'].map('${:,.2f}'.format)
            df_e['Mean ($)'] = df_e['Mean ($)'].map('${:,.2f}'.format)
            st.dataframe(df_e, use_container_width=True)

# ------------------------------------------------------------------------------
# 6. View 3: Task 3: Business Insights
# ------------------------------------------------------------------------------
elif nav_selection == "Task 3: Business Insights":
    st.markdown("""
    <div class="main-header">
        <h1>Task 3: Executive Business Questions</h1>
        <p>In-depth analytical findings addressing salary outliers (Q1), managerial wage disparities (Q2), and historical hiring trends (Q3).</p>
    </div>
    """, unsafe_allow_html=True)

    if app_data is None:
        st.error("Data cache not found.")
        st.stop()

    t3_tab1, t3_tab2, t3_tab3 = st.tabs([
        "Q1: Compensation Outliers",
        "Q2: Manager vs Employee Disparities",
        "Q3: Historical Hiring Cohorts"
    ])

    with t3_tab1:
        st.markdown("### Q1: Salary Outliers & Extreme Compensations")
        q1_data = app_data['task3_q1']
        
        col_q1_1, col_q1_2 = st.columns(2)
        with col_q1_1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">IQR Upper Outliers Flagged</div>
                <div class="metric-value">{q1_data['iqr_count']:,} ({q1_data['iqr_pct']}%)</div>
                <div class="metric-sub">Cutoff Fence: > ${q1_data['iqr_upper']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_q1_2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Z-Score Outliers (|Z| > 3.0)</div>
                <div class="metric-value">{q1_data['z_count']:,} ({q1_data['z_pct']}%)</div>
                <div class="metric-sub">Cutoff Threshold: ≥ ${q1_data['z_upper']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Top 10 Highest Public Compensations in Texas (Sanitized)")
        df_top = pd.DataFrame(q1_data['top_10_highest'])
        df_top.columns = ['Position Title', 'Agency Name', 'Annual Salary', 'Status', 'Tenure (Yrs)']
        df_top['Annual Salary'] = df_top['Annual Salary'].map('${:,.2f}'.format)
        st.dataframe(df_top, use_container_width=True)

        st.markdown("#### Top 10 Lowest Compensations (Part-Time & Intern Positions)")
        df_low = pd.DataFrame(q1_data['top_10_lowest'])
        df_low.columns = ['Position Title', 'Agency Name', 'Annual Salary', 'Hrs/Wk', 'Status']
        df_low['Annual Salary'] = df_low['Annual Salary'].map('${:,.2f}'.format)
        st.dataframe(df_low, use_container_width=True)

    with t3_tab2:
        st.markdown("### Q2: Manager vs Regular Employee Wage Disparities")
        q2_data = app_data['task3_q2']
        
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Manager Median Salary</div>
                <div class="metric-value">${q2_data['state_mgr_median']:,.0f}</div>
                <div class="metric-sub">8,641 Managerial Positions (5.8%)</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Employee Median Salary</div>
                <div class="metric-value">${q2_data['state_emp_median']:,.0f}</div>
                <div class="metric-sub">140,840 Frontline Personnel</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Statewide Disparity Ratio</div>
                <div class="metric-value">{q2_data['state_ratio']}x</div>
                <div class="metric-sub">Managers earn 96% more on average</div>
            </div>
            """, unsafe_allow_html=True)

        c_d1, c_d2 = st.columns(2)
        with c_d1:
            st.markdown("#### Top 10 Highest Wage Disparity Agencies")
            df_td = pd.DataFrame(q2_data['top_disparity'])[['agency', 'mgr_median', 'emp_median', 'ratio', 'gap']]
            df_td.columns = ['Agency', 'Mgr Med ($)', 'Emp Med ($)', 'Ratio', 'Gap ($)']
            df_td['Mgr Med ($)'] = df_td['Mgr Med ($)'].map('${:,.0f}'.format)
            df_td['Emp Med ($)'] = df_td['Emp Med ($)'].map('${:,.0f}'.format)
            df_td['Gap ($)'] = df_td['Gap ($)'].map('${:,.0f}'.format)
            df_td['Ratio'] = df_td['Ratio'].map('{:.2f}x'.format)
            st.dataframe(df_td, use_container_width=True)

        with c_d2:
            st.markdown("#### Top 10 Most Egalitarian Agencies (Lowest Disparity)")
            df_bd = pd.DataFrame(q2_data['bottom_disparity'])[['agency', 'mgr_median', 'emp_median', 'ratio', 'gap']]
            df_bd.columns = ['Agency', 'Mgr Med ($)', 'Emp Med ($)', 'Ratio', 'Gap ($)']
            df_bd['Mgr Med ($)'] = df_bd['Mgr Med ($)'].map('${:,.0f}'.format)
            df_bd['Emp Med ($)'] = df_bd['Emp Med ($)'].map('${:,.0f}'.format)
            df_bd['Gap ($)'] = df_bd['Gap ($)'].map('${:,.0f}'.format)
            df_bd['Ratio'] = df_bd['Ratio'].map('{:.2f}x'.format)
            st.dataframe(df_bd, use_container_width=True)

    with t3_tab3:
        st.markdown("### Q3: Historical Hiring Cohort Trends (1980–2020)")
        q3_data = app_data['task3_q3']
        df_co = pd.DataFrame(q3_data['recent_cohorts'])
        df_co.columns = ['Hire Year', 'Headcount', 'Median Salary ($)', 'Mean Salary ($)']

        fig, ax1 = plt.subplots(figsize=(10, 4.2))
        ax2 = ax1.twinx()
        
        ax1.bar(df_co['Hire Year'], df_co['Headcount'], color='#93c5fd', alpha=0.7, label='Cohort Headcount')
        ax2.plot(df_co['Hire Year'], df_co['Median Salary ($)'], color='#1e3a8a', marker='o', linewidth=2.5, label='Median Salary')
        
        ax1.set_xlabel("Hiring Cohort Year", fontsize=10)
        ax1.set_ylabel("Hired Headcount", color='#1e40af', fontsize=10)
        ax2.set_ylabel("Median Salary ($)", color='#1e3a8a', fontsize=10)
        ax2.yaxis.set_major_formatter('${x:,.0f}')
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

        st.info("**Survivorship Bias Caution:** In single-snapshot administrative data, earlier cohorts exhibit higher compensation due to tenured survivor bias (veteran workers who stayed and promoted into management).")

# ------------------------------------------------------------------------------
# 7. View 4: Model Benchmarks (Task 2)
# ------------------------------------------------------------------------------
elif nav_selection == "Model Benchmarks (Task 2)":
    st.markdown("""
    <div class="main-header">
        <h1>Model Benchmarking & Architectural Evaluation (Task 2)</h1>
        <p>Comparative empirical results across all 9 regression algorithms evaluated on unseen test data (29,895 records).</p>
    </div>
    """, unsafe_allow_html=True)

    if app_data is None:
        st.error("Data cache not found.")
        st.stop()

    bench_df = pd.DataFrame(app_data['benchmarks'])
    bench_display = bench_df.copy()
    bench_display['Train R2'] = bench_display['Train R2'].map('{:.4f}'.format)
    bench_display['Test R2'] = bench_display['Test R2'].map('{:.4f}'.format)
    bench_display['Adj R2'] = bench_display['Adj R2'].map('{:.4f}'.format)
    bench_display['MAE'] = bench_display['MAE'].map('${:,.2f}'.format)
    bench_display['RMSE'] = bench_display['RMSE'].map('${:,.2f}'.format)
    bench_display['MAPE'] = bench_display['MAPE'].map('{:.2f}%'.format)
    bench_display['Fit Time (s)'] = bench_display['Fit Time (s)'].map('{:.2f}s'.format)

    st.markdown("### Master Performance Matrix")
    st.dataframe(bench_display, use_container_width=True)

    st.markdown("### Test R² and MAE Comparison")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    order_r2 = bench_df.sort_values('Test R2', ascending=True)
    ax1.barh(order_r2['Model'], order_r2['Test R2'], color='#1e40af')
    ax1.set_title("Test R² (Higher is Better)", fontweight='bold')
    ax1.set_xlim(0, 1.0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    order_mae = bench_df.sort_values('MAE', ascending=False)
    ax2.barh(order_mae['Model'], order_mae['MAE'], color='#e11d48')
    ax2.set_title("Test MAE in $ USD (Lower is Better)", fontweight='bold')
    ax2.xaxis.set_major_formatter('${x:,.0f}')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.success("**Final Production Choice:** Tuned LightGBM Regressor delivers the optimal balance of maximum accuracy ($R^2 = 0.9518$, MAE = $2,982.79), minimal overfitting gap ($+0.0065$), rapid training latency (3.53s), and compact artifact size (3.24 MB).")


