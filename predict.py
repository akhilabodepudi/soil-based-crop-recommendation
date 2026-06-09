"""Predict the best crop for a given soil/weather profile.

Examples
--------
    python predict.py --N 90 --P 42 --K 43 --temperature 20.8 \
        --humidity 82 --ph 6.5 --rainfall 202

    python predict.py --model models/XGBoost.pkl --N 104 --P 18 --K 30 \
        --temperature 23.6 --humidity 60.3 --ph 6.7 --rainfall 140.9
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_model(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend a crop from soil/weather inputs.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "RandomForest.pkl"),
        help="Path to a trained .pkl model (default: models/RandomForest.pkl)",
    )
    parser.add_argument("--N", type=float, required=True, help="Nitrogen content in soil")
    parser.add_argument("--P", type=float, required=True, help="Phosphorus content in soil")
    parser.add_argument("--K", type=float, required=True, help="Potassium content in soil")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature (C)")
    parser.add_argument("--humidity", type=float, required=True, help="Relative humidity (%)")
    parser.add_argument("--ph", type=float, required=True, help="Soil pH value")
    parser.add_argument("--rainfall", type=float, required=True, help="Rainfall (mm)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}\nRun `python train.py` first.")

    model = load_model(model_path)
    sample = pd.DataFrame(
        [[args.N, args.P, args.K, args.temperature, args.humidity, args.ph, args.rainfall]],
        columns=FEATURES,
    )

    prediction = model.predict(sample)[0]

    # XGBoost was trained on integer-encoded labels; map back to the crop name.
    if not isinstance(prediction, str):
        encoder = LabelEncoder().fit(pd.read_csv(DATA_PATH)["label"])
        prediction = encoder.inverse_transform([int(prediction)])[0]

    print(f"Recommended crop: {prediction}")


if __name__ == "__main__":
    main()
