# 🌾 Soil-Based Crop Recommendation

A machine-learning project that recommends the **most suitable crop** to cultivate based on
soil nutrients and local weather conditions. It trains and compares **six classification
algorithms** on the classic crop-recommendation dataset and ships the trained models plus
ready-to-use training and prediction scripts.

This is the ML engine behind the [AgroNity](https://github.com/akhilabodepudi/AgroNity)
digital farming platform.

---

## 📌 Problem

Given a soil and climate profile, predict which of **22 crops** will grow best:

| Feature | Description | Unit |
| --- | --- | --- |
| `N` | Nitrogen content in soil | ratio |
| `P` | Phosphorus content in soil | ratio |
| `K` | Potassium content in soil | ratio |
| `temperature` | Average temperature | °C |
| `humidity` | Relative humidity | % |
| `ph` | Soil pH value | 0–14 |
| `rainfall` | Rainfall | mm |

**Target:** one of 22 crops — apple, banana, blackgram, chickpea, coconut, coffee, cotton,
grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange,
papaya, pigeonpeas, pomegranate, rice, watermelon.

---

## 📊 Dataset

- **2,200 samples** — 100 per crop (perfectly balanced)
- **7 numeric features** + 1 label column
- File: [`data/crop_recommendation.csv`](data/crop_recommendation.csv)

---

## 🤖 Models & Results

Six classifiers were trained on an 80/20 train–test split and compared by accuracy on the
hold-out test set:

| Model | Test Accuracy |
| --- | :---: |
| **XGBoost** | **99.55%** |
| Gaussian Naive Bayes | 99.09% |
| Random Forest | 99.09% |
| Support Vector Machine | 97.95% |
| Logistic Regression | 95.23% |
| Decision Tree | 90.00% |

Each trained model is saved under [`models/`](models/) as a pickle file.

> The full exploratory analysis, training, evaluation, and accuracy comparison live in
> [`notebooks/crop_recommendation_analysis.ipynb`](notebooks/crop_recommendation_analysis.ipynb).

---

## 🗂️ Project Structure

```text
soil-based-crop-recommendation/
├── data/
│   └── crop_recommendation.csv          # 2,200-row dataset
├── models/                              # Trained, pickled classifiers
│   ├── DecisionTree.pkl
│   ├── NBClassifier.pkl
│   ├── SVMClassifier.pkl
│   ├── LogisticRegression.pkl
│   ├── RandomForest.pkl
│   └── XGBoost.pkl
├── notebooks/
│   └── crop_recommendation_analysis.ipynb
├── train.py                             # Train & save all six models
├── predict.py                           # Recommend a crop from inputs
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Setup

```bash
git clone https://github.com/akhilabodepudi/soil-based-crop-recommendation.git
cd soil-based-crop-recommendation

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Make a prediction (uses the included models)

```bash
python predict.py --N 90 --P 42 --K 43 \
    --temperature 20.8 --humidity 82 --ph 6.5 --rainfall 202
# Recommended crop: rice
```

Pick a different model with `--model`:

```bash
python predict.py --model models/XGBoost.pkl --N 104 --P 18 --K 30 \
    --temperature 23.6 --humidity 60.3 --ph 6.7 --rainfall 140.9
```

### 3. Re-train the models

```bash
python train.py
```

This retrains all six classifiers and overwrites the pickles in `models/`.

---

## 🧰 Tech Stack

- **Python**, **pandas**, **NumPy**
- **scikit-learn** (Decision Tree, Naive Bayes, SVM, Logistic Regression, Random Forest)
- **XGBoost**
- **Matplotlib** (accuracy comparison plot)
- **Jupyter Notebook**

---

## 🛣️ Future Work

- Hyperparameter tuning and probability-calibrated outputs
- Confidence scores and top-3 crop suggestions
- Expand features with real-time weather and soil-pollution data
- Serve predictions via a REST API for the AgroNity platform

---

## 📚 References

- Crop Recommendation Dataset (N, P, K, temperature, humidity, pH, rainfall)
- scikit-learn and XGBoost documentation
