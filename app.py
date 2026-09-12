import streamlit as st
import joblib
import pandas as pd
import shap
import numpy as np


def calculate_psi(reference, current, bins=10):
    """
    Calculate Population Stability Index (PSI).

    reference: Training/reference data
    current: New/current data
    """

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    # Create bins using reference data quantiles
    breakpoints = np.quantile(
        reference,
        np.linspace(0, 1, bins + 1)
    )

    # Remove duplicate breakpoints
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        return 0.0

    # Include all possible values
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_bins = pd.cut(
        reference,
        bins=breakpoints,
        include_lowest=True
    )

    current_bins = pd.cut(
        current,
        bins=breakpoints,
        include_lowest=True
    )

    reference_distribution = (
        reference_bins
        .value_counts(sort=False, normalize=True)
    )

    current_distribution = (
        current_bins
        .value_counts(sort=False, normalize=True)
    )

    # Avoid division by zero
    reference_distribution = np.clip(
        reference_distribution,
        0.000001,
        None
    )

    current_distribution = np.clip(
        current_distribution,
        0.000001,
        None
    )

    psi = np.sum(
        (current_distribution - reference_distribution)
        * np.log(
            current_distribution / reference_distribution
        )
    )

    return float(psi)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Fraud Detection & Risk Analysis",
    page_icon="💳",
    layout="wide"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("fraud_pipeline.pkl")


model = load_model()


# ==========================================
# HEADER
# ==========================================

st.title("💳 Fraud Detection & Risk Analysis")

st.write(
    "Machine-learning powered transaction "
    "risk assessment system."
)

st.divider()


# ==========================================
# FILE UPLOAD
# ==========================================

st.subheader("📂 Upload Transaction Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing transaction data",
    type=["csv"]
)


# ==========================================
# PROCESS FILE
# ==========================================

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Transactions")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )


    # ======================================
    # REQUIRED FEATURES
    # ======================================

    required_features = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount"
    ]


    # ======================================
    # CHECK FEATURES
    # ======================================

    missing_features = [
        feature
        for feature in required_features
        if feature not in data.columns
    ]


    if missing_features:

        st.error(
            "The uploaded file is missing "
            "required features."
        )

        st.write(missing_features)


    else:

        # ==================================
        # MODEL PREDICTION
        # ==================================

        X = data[required_features]

        probabilities = model.predict_proba(X)[:, 1]

        risk_scores = probabilities * 100


        # ==================================
        # RISK LEVEL FUNCTION
        # ==================================

        def assign_risk_level(score):

            if score < 20:
                return "Low"

            elif score < 50:
                return "Medium"

            elif score < 80:
                return "High"

            else:
                return "Critical"


        risk_levels = [
            assign_risk_level(score)
            for score in risk_scores
        ]


        # ==================================
        # RESULTS DATAFRAME
        # ==================================

        results = data.copy()

        results["Fraud_Probability"] = probabilities

        results["Risk_Score"] = risk_scores

        results["Risk_Level"] = risk_levels


        # ==================================
        # DASHBOARD METRICS
        # ==================================

        st.subheader("📈 Risk Overview")

        total_transactions = len(results)

        high_risk_count = (
            results["Risk_Level"]
            .isin(["High", "Critical"])
            .sum()
        )

        critical_count = (
            results["Risk_Level"] == "Critical"
        ).sum()

        average_risk = (
            results["Risk_Score"].mean()
        )


        col1, col2, col3, col4 = st.columns(4)


        col1.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )


        col2.metric(
            "High/Critical Risk",
            f"{high_risk_count:,}"
        )


        col3.metric(
            "Critical Transactions",
            f"{critical_count:,}"
        )


        col4.metric(
            "Average Risk Score",
            f"{average_risk:.2f}"
        )


        st.divider()


        # ==================================
        # RISK DISTRIBUTION
        # ==================================

        st.subheader("🚨 Risk Level Distribution")

        risk_distribution = (
            results["Risk_Level"]
            .value_counts()
            .reindex(
                [
                    "Low",
                    "Medium",
                    "High",
                    "Critical"
                ],
                fill_value=0
            )
        )

        st.bar_chart(risk_distribution)


        # ==================================
        # TRANSACTION EXPLORER
        # ==================================

        st.subheader("🔎 Transaction Explorer")

        selected_risk = st.selectbox(
            "Filter by Risk Level",
            [
                "All",
                "Low",
                "Medium",
                "High",
                "Critical"
            ]
        )


        if selected_risk == "All":

            filtered_results = results

        else:

            filtered_results = results[
                results["Risk_Level"]
                == selected_risk
            ]


        # ==================================
        # RESULTS TABLE
        # ==================================

        display_columns = [
            "Fraud_Probability",
            "Risk_Score",
            "Risk_Level"
        ]


        if "Amount" in results.columns:

            display_columns.insert(
                0,
                "Amount"
            )


        st.dataframe(
            filtered_results[display_columns],
            use_container_width=True
        )


        st.write(
            f"Showing **{len(filtered_results):,} "
            f"transactions**."
        )

# ==========================================
# SHAP EXPLAINABILITY
# ==========================================

st.divider()

st.subheader("🔍 Explain a Transaction")

st.write(
    "Select a transaction to understand which "
    "features influenced its fraud-risk prediction."
)

# Select transaction index
selected_index = st.number_input(
    "Transaction Index",
    min_value=0,
    max_value=len(results) - 1,
    value=0,
    step=1
)

if st.button("Explain Transaction"):

    # Select the transaction
    selected_transaction = X.iloc[[selected_index]]

    # Access the trained XGBoost model
    xgb_model = model.named_steps["xgb"]

    # Transform the selected transaction
    transformed_transaction = model.named_steps[
        "preprocessor"
    ].transform(selected_transaction)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(
        transformed_transaction
    )

    st.write("### SHAP Feature Contributions")

    shap_df = pd.DataFrame({
        "Feature": required_features,
        "SHAP_Value": shap_values[0]
    })

    shap_df["Absolute_SHAP"] = np.abs(
        shap_df["SHAP_Value"]
    )

    shap_df = shap_df.sort_values(
        by="Absolute_SHAP",
        ascending=False
    )

    st.dataframe(
        shap_df[
            [
                "Feature",
                "SHAP_Value"
            ]
        ].head(10),
        use_container_width=True
    )


# --------------------------------------------------
# Data Drift Monitoring
# --------------------------------------------------

st.divider()

st.subheader("📡 Data Drift Monitoring")

st.write(
    "This section compares the uploaded transactions "
    "with the original training dataset using PSI."
)

# Load original training data
reference_df = pd.read_csv("data/creditcard.csv")

if len(X) < 100:
    st.warning(
        "The uploaded dataset contains fewer than 100 transactions. "
        "PSI results may be unstable for very small datasets."
    )

drift_features = [
    "Time",
    "Amount",
    "V1",
    "V2",
    "V3",
    "V10",
    "V14"
]

drift_results = []

for feature in drift_features:
    if feature in reference_df.columns and feature in X.columns:
        psi_value = calculate_psi(
            reference_df[feature],
            X[feature]
        )

        if psi_value < 0.10:
            status = "No Significant Drift"
        elif psi_value <= 0.25:
            status = "Moderate Drift"
        else:
            status = "Significant Drift"

        drift_results.append({
            "Feature": feature,
            "PSI": round(psi_value, 4),
            "Status": status
        })

drift_df = pd.DataFrame(drift_results)

st.dataframe(
    drift_df,
    use_container_width=True
)

st.info(
    "PSI values above 0.25 indicate significant distribution change "
    "and may require investigation or model retraining."
)

