import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def generate_synthetic_ehr(n_samples=5000):
    """Generates synthetic structured EHR patient dataset."""
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 90, n_samples),
        'length_of_stay_days': np.random.poisson(lam=5, size=n_samples) + 1,
        'num_lab_procedures': np.random.randint(1, 100, n_samples),
        'num_medications': np.random.randint(1, 30, n_samples),
        'num_emergency_visits_prior_year': np.random.poisson(lam=1, size=n_samples),
        'has_diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'has_hypertension': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'readmitted_30d': np.random.choice([0, 1], n_samples, p=[0.82, 0.18])
    }
    return pd.DataFrame(data)

def train_readmission_model():
    """Trains XGBoost model and calculates SHAP values."""
    df = generate_synthetic_ehr()
    X = df.drop(columns=['readmitted_30d'])
    y = df['readmitted_30d']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale positive weights for class imbalance
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    model = xgb.XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    print(classification_report(y_test, preds))
    
    # Compute SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    return model, explainer, shap_values, X_test

if __name__ == "__main__":
    train_readmission_model()