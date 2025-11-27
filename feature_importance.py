import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("clean_telco.csv")

# Features & Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost model
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1])
)
xgb.fit(X_train, y_train)

# Feature importance
importances = xgb.feature_importances_
features = X.columns
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Normalize importance
importance_df["Importance"] = importance_df["Importance"] / importance_df["Importance"].sum()

# Save all features to CSV
importance_df.to_csv("feature_importance.csv", index=False)
print("✅ Feature importance saved to feature_importance.csv")

# Select top 10 for plotting
top_10 = importance_df.head(10)

# Plot
plt.figure(figsize=(10,6))
sns.barplot(
    x="Importance", 
    y="Feature", 
    hue="Feature",       
    data=top_10,
    palette="viridis",
    legend=False      
)

# Add percentage labels
for i, val in enumerate(top_10["Importance"]):
    plt.text(val + 0.002, i, f"{val:.2%}", va="center")

plt.title("Top 10 Churn Drivers (XGBoost Feature Importance)", fontsize=14, weight="bold")
plt.xlabel("Relative Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()