# Kaggle experiments

Small, self-contained machine learning exercises based on popular Kaggle datasets. Each folder contains a single Python script; datasets are intentionally not committed.

## Project map
- `Breast-Cancer-Classification/BC_Classifier.py` — Logistic regression on `sklearn.datasets.load_breast_cancer()` with a simple prediction example.
- `Car-Price-Prediction/Car_Price_Predictor.py` — Linear and Lasso regression to estimate selling price. Expects `car data.csv` from the Kaggle car price dataset in the same folder.
- `Coaster-Analysis/Data-Analysis.py` — Exploratory data analysis of coaster stats. Expects `coaster_db.csv` from the Kaggle rollercoaster dataset in the same folder.
- `Credit-Card-Fraud-Detection/Fraud_Detection.py` — Logistic regression with undersampling. Expects `creditcard.csv` from the Kaggle credit card fraud dataset in the same folder.

## Setup
1. From this folder, create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install shared dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Download the required CSVs from Kaggle and place each file in its matching project folder (names listed above).

## Running a script
Activate the virtual environment and execute the relevant file, e.g.:
```bash
python Breast-Cancer-Classification/BC_Classifier.py
python Car-Price-Prediction/Car_Price_Predictor.py
python Coaster-Analysis/Data-Analysis.py
python Credit-Card-Fraud-Detection/Fraud_Detection.py
```
Uncomment the exploratory print/plot sections inside each script if you want to inspect the data or visualize results.
