"""Train and compare crop-recommendation models.

Trains six classifiers on the crop-recommendation dataset and saves each one
as a pickle file in ``models/``. Run from the project root:

    python train.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "crop_recommendation.csv"
MODELS_DIR = ROOT / "models"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
RANDOM_STATE = 2


def load_data(path: Path = DATA_PATH):
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df["label"]
    return X, y


def save_model(model, filename: str) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / filename, "wb") as fh:
        pickle.dump(model, fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crop-recommendation models.")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test split fraction (default: 0.2)"
    )
    args = parser.parse_args()

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE
    )

    scores: dict[str, float] = {}

    # --- Decision Tree ---
    dt = DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE, max_depth=5)
    dt.fit(X_train, y_train)
    scores["DecisionTree"] = accuracy_score(y_test, dt.predict(X_test))
    save_model(dt, "DecisionTree.pkl")

    # --- Gaussian Naive Bayes ---
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    scores["NaiveBayes"] = accuracy_score(y_test, nb.predict(X_test))
    save_model(nb, "NBClassifier.pkl")

    # --- Support Vector Machine (scaled) ---
    scaler = MinMaxScaler().fit(X_train)
    svm = SVC(kernel="poly", degree=3, C=1)
    svm.fit(scaler.transform(X_train), y_train)
    scores["SVM"] = accuracy_score(y_test, svm.predict(scaler.transform(X_test)))
    save_model(svm, "SVMClassifier.pkl")

    # --- Logistic Regression ---
    logreg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    logreg.fit(X_train, y_train)
    scores["LogisticRegression"] = accuracy_score(y_test, logreg.predict(X_test))
    save_model(logreg, "LogisticRegression.pkl")

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=20, random_state=0)
    rf.fit(X_train, y_train)
    scores["RandomForest"] = accuracy_score(y_test, rf.predict(X_test))
    save_model(rf, "RandomForest.pkl")

    # --- XGBoost (labels must be encoded as integers) ---
    import xgboost as xgb

    encoder = LabelEncoder().fit(y)
    xb = xgb.XGBClassifier()
    xb.fit(X_train, encoder.transform(y_train))
    xb_pred = encoder.inverse_transform(xb.predict(X_test))
    scores["XGBoost"] = accuracy_score(y_test, xb_pred)
    save_model(xb, "XGBoost.pkl")

    print("\nModel accuracy on the hold-out test set")
    print("-" * 40)
    for name, acc in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{name:<20} {acc * 100:6.2f}%")
    print(f"\nModels saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
