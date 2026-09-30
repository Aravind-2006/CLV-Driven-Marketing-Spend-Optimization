import os
import datetime
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# Import business logic layer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.business_layer import evaluate_customer_spend, HIGH_SPEND, MEDIUM_SPEND, LOW_SPEND, LOW_THRESHOLD, HIGH_THRESHOLD

# Page config
st.set_page_config(
    page_title="CLV Marketing Optimization Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">MLOps Customer Lifetime Value (CLV) & Spend Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Domain: Retail & E-Commerce | SDG 8: Decent Work and Economic Growth | IT4V43 MLOps PBL</div>', unsafe_allow_html=True)

# Helper functions to load data and model
@st.cache_data
def load_rfm_data():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "rfm_features.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "model.json")
    if os.path.exists(model_path):
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return model, os.path.getmtime(model_path)
    return None, None

df_rfm = load_rfm_data()
model, model_mtime = load_trained_model()

# Tabs
tab1, tab2 = st.tabs(["📊 Technical MLOps Panel", "💼 Business Translation & Spend Panel"])

# ==========================================
# TAB 1: TECHNICAL PANEL
# ==========================================
with tab1:
    st.header("Technical Model Performance & Pipeline Verification")
    
    if df_rfm is not None and model is not None:
        # Re-evaluate test set metrics
        X = df_rfm[["Recency", "Frequency"]]
        y_log = np.log1p(df_rfm["Monetary"])
        X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
        
        preds_log = model.predict(X_test)
        preds_actual = np.expm1(preds_log)
        y_test_actual = np.expm1(y_test_log)
        
        mae = mean_absolute_error(y_test_actual, preds_actual)
        r2 = r2_score(y_test_log, preds_log)
        
        model_time_str = datetime.datetime.fromtimestamp(model_mtime).strftime("%Y-%m-%d %H:%M:%S") if model_mtime else "N/A"
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Model Architecture", value="XGBoost Regressor")
        col2.metric(label="Log-Target R² Score", value=f"{r2:.4f}")
        col3.metric(label="Test MAE (₹)", value=f"₹{mae:,.2f}")
        col4.metric(label="Last Trained Timestamp", value=model_time_str)
        
        st.markdown("---")
        st.subheader("Actual vs. Predicted Monetary Value (Test Set Sample)")
        
        test_df = pd.DataFrame({
            "Actual Monetary (₹)": y_test_actual.values,
            "Predicted Monetary (₹)": preds_actual
        })
        
        # Display sample comparison chart
        sample_df = test_df.sample(n=min(50, len(test_df)), random_state=42).sort_values(by="Actual Monetary (₹)")
        st.bar_chart(sample_df.set_index(np.arange(len(sample_df))), height=350)
        
        with st.expander("View Sample Prediction Table"):
            st.dataframe(sample_df.reset_index(drop=True).style.format({"Actual Monetary (₹)": "₹{:,.2f}", "Predicted Monetary (₹)": "₹{:,.2f}"}))

    else:
        st.error("Model file (`model.json`) or data (`rfm_features.csv`) not found. Please run `train.py` first.")

# ==========================================
# TAB 2: BUSINESS PANEL
# ==========================================
with tab2:
    st.header("Business Cost-Translation & Marketing Spend Optimization")
    
    if df_rfm is not None and model is not None:
        # Segment all customers
        X_all = df_rfm[["Recency", "Frequency"]]
        preds_log_all = model.predict(X_all)
        preds_all = np.expm1(preds_log_all)
        
        evaluations = [evaluate_customer_spend(p) for p in preds_all]
        eval_df = pd.DataFrame(evaluations)
        
        combined_df = pd.concat([df_rfm, eval_df], axis=1)
        
        # Aggregate Segment Statistics
        seg_counts = combined_df["segment"].value_counts()
        total_customers = len(combined_df)
        total_targeted_spend = combined_df["allocated_spend"].sum()
        
        # Naive Baseline: Flat ₹100 spend for every customer
        FLAT_BASELINE_PER_CUST = 100.0
        total_naive_spend = total_customers * FLAT_BASELINE_PER_CUST
        cost_difference = total_naive_spend - total_targeted_spend
        
        # Summary KPI Cards
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        b_col1.metric(label="Total Customer Base", value=f"{total_customers:,}")
        b_col2.metric(label="Targeted Spend Allocated", value=f"₹{total_targeted_spend:,.2f}")
        b_col3.metric(label="Naive Baseline Spend (₹100/cust)", value=f"₹{total_naive_spend:,.2f}")
        b_col4.metric(
            label="Waste Avoided on Low-Value Segment",
            value=f"₹{(combined_df['segment'] == 'Low').sum() * FLAT_BASELINE_PER_CUST:,.2f}",
            delta="Zero budget wasted on churned/low-value"
        )
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns([1, 1])
        
        with col_chart1:
            st.subheader("Customer Distribution by CLV Segment")
            chart_data = pd.DataFrame({
                "Segment": seg_counts.index,
                "Customer Count": seg_counts.values
            }).set_index("Segment")
            st.bar_chart(chart_data)
            
        with col_chart2:
            st.subheader("Spend Allocation Rules & Thresholds")
            st.markdown(f"""
            - **High Value Segment** (`CLV >= ₹{HIGH_THRESHOLD:,.2f}`):  
              Allocation: **₹{HIGH_SPEND:.0f} / month** | *VIP Retention & Upsell*
            - **Medium Value Segment** (`₹{LOW_THRESHOLD:,.2f} <= CLV < ₹{HIGH_THRESHOLD:,.2f}`):  
              Allocation: **₹{MEDIUM_SPEND:.0f} / month** | *Nurturing & Re-engagement*
            - **Low Value Segment** (`CLV < ₹{LOW_THRESHOLD:,.2f}`):  
              Allocation: **₹{LOW_SPEND:.0f} / month** | *Zero Budget Waste*
            """)
            
        st.markdown("---")
        st.subheader("🎯 Live Customer CLV & Spend Allocation Calculator")
        st.write("Enter hypothetical customer purchase behavior to receive immediate CLV prediction and spend recommendation:")
        
        calc_col1, calc_col2 = st.columns(2)
        with calc_col1:
            input_recency = st.slider("Recency (Days since last purchase):", min_value=1, max_value=365, value=30)
        with calc_col2:
            input_frequency = st.slider("Frequency (Total number of purchase orders):", min_value=1, max_value=100, value=5)
            
        input_data = np.array([[input_recency, input_frequency]])
        pred_log = model.predict(input_data)[0]
        pred_val = float(np.expm1(pred_log))
        
        res = evaluate_customer_spend(pred_val)
        
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("Predicted CLV (Monetary)", f"₹{res['predicted_monetary_value']:,.2f}")
        res_col2.metric("Customer Segment", res['segment'])
        res_col3.metric("Allocated Monthly Spend", f"₹{res['allocated_spend']:.2f}")
        res_col4.metric("Cost Efficiency ROI Ratio", f"{res['cost_efficiency_ratio']}x")

    else:
        st.error("Model file (`model.json`) or data (`rfm_features.csv`) not found. Please run `train.py` first.")
