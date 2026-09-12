import numpy as np
import pandas as pd
import joblib


# --------------------------------------------------
# Test 1: Required features
# --------------------------------------------------

def test_required_features():
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

    df = pd.read_csv("data/test_transactions.csv")

    assert set(required_features).issubset(df.columns)


# --------------------------------------------------
# Test 2: Dataset should not be empty
# --------------------------------------------------

def test_dataset_not_empty():
    df = pd.read_csv("data/test_transactions.csv")

    assert len(df) > 0


# --------------------------------------------------
# Test 3: Risk scores should be between 0 and 100
# --------------------------------------------------

def test_risk_score_range():
    probabilities = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    risk_scores = probabilities * 100

    assert np.all(risk_scores >= 0)
    assert np.all(risk_scores <= 100)


# --------------------------------------------------
# Test 4: Risk categories
# --------------------------------------------------

def assign_risk_category(score):
    if score < 20:
        return "Low"
    elif score < 50:
        return "Medium"
    elif score < 80:
        return "High"
    else:
        return "Critical"


def test_risk_categories():
    assert assign_risk_category(10) == "Low"
    assert assign_risk_category(30) == "Medium"
    assert assign_risk_category(60) == "High"
    assert assign_risk_category(90) == "Critical"


# --------------------------------------------------
# Test 5: PSI calculation
# --------------------------------------------------

def calculate_psi(reference, current, bins=10):
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    breakpoints = np.quantile(
        reference,
        np.linspace(0, 1, bins + 1)
    )

    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        return 0.0

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


def test_psi_no_drift():
    reference = np.arange(1, 101)
    current = np.arange(1, 101)

    psi_value = calculate_psi(reference, current)

    assert psi_value < 0.10


def test_psi_detects_drift():
    reference = np.arange(1, 101)
    current = np.arange(1000, 1100)

    psi_value = calculate_psi(reference, current)

    assert psi_value > 0.10


# --------------------------------------------------
# Test 6: Model prediction
# --------------------------------------------------

def test_model_prediction():
    model = joblib.load("fraud_pipeline.pkl")

    df = pd.read_csv("data/test_transactions.csv")

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

    X = df[required_features]

    predictions = model.predict_proba(X)[:, 1]

    assert len(predictions) == len(X)
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)

    