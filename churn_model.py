import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("clean_telco.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    random_state=42,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
    eval_metric="logloss"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


orig = pd.read_csv("churn_model.csv")
test_ids = orig.iloc[y_test.index]["customerID"].values

results = pd.DataFrame({
    "CustomerID": test_ids,
    "ActualChurn": y_test.values,
    "PredictedLabel": y_pred,
    "ChurnProbability": y_proba.round(4)
})

results.to_csv("churn_predictions.csv", index=False)
print("Predictions with probabilities exported to churn_predictions.csv")
print(results.head())