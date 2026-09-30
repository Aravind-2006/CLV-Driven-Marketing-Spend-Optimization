import os
import pandas as pd

# Marketing spend constants per customer segment (in INR / month)
HIGH_SPEND = 500.0
MEDIUM_SPEND = 100.0
LOW_SPEND = 0.0


def load_segment_thresholds(csv_path="data/rfm_features.csv"):
    """
    Computes percentile-based segment thresholds dynamically from the RFM dataset.
    - Low threshold: 25th percentile of Monetary value (~307.25)
    - High threshold: 75th percentile of Monetary value (~1661.64)
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        low_thresh = float(df["Monetary"].quantile(0.25))
        high_thresh = float(df["Monetary"].quantile(0.75))
    else:
        # Fallback thresholds based on UCI Online Retail RFM analysis
        low_thresh = 307.25
        high_thresh = 1661.64

    return low_thresh, high_thresh


# Pre-load default thresholds
LOW_THRESHOLD, HIGH_THRESHOLD = load_segment_thresholds()


def evaluate_customer_spend(predicted_monetary_value: float):
    """
    Translates predicted CLV (monetary value) into customer segment,
    marketing spend allocation, and cost-efficiency metric (ROI ratio).
    """
    if predicted_monetary_value >= HIGH_THRESHOLD:
        segment = "High"
        allocated_spend = HIGH_SPEND
    elif predicted_monetary_value >= LOW_THRESHOLD:
        segment = "Medium"
        allocated_spend = MEDIUM_SPEND
    else:
        segment = "Low"
        allocated_spend = LOW_SPEND

    if allocated_spend > 0:
        cost_efficiency_ratio = round(predicted_monetary_value / allocated_spend, 2)
    else:
        cost_efficiency_ratio = 0.0

    return {
        "predicted_monetary_value": round(float(predicted_monetary_value), 2),
        "segment": segment,
        "allocated_spend": allocated_spend,
        "cost_efficiency_ratio": cost_efficiency_ratio,
    }
