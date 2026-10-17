
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load dataset
df = pd.read_csv("dispensing.csv", parse_dates=["date"])
df = df.sort_values(["store_id", "drug_id", "date"])

# 2. Create next-week stock-out target
df["target_next_week"] = (
    df.groupby(["store_id", "drug_id"])["stockout"].shift(-1)
)

df = df.dropna(subset=["target_next_week"]).copy()
df["target_next_week"] = df["target_next_week"].astype(int)

# 3. Select features available in the current record
numeric_features = [
    "week",
    "demand",
    "units_dispensed",
    "stock_on_hand",
    "units_ordered",
    "unit_price_inr"
]

categorical_features = [
    "store_id",
    "drug_id",
    "region",
    "patient_age_group",
    "copay_type",
    "prescriber_specialty",
    "dispense_status"
]

features = numeric_features + categorical_features
target = "target_next_week"

# 4. Split chronologically
split_date = df["date"].quantile(0.8)
train = df[df["date"] < split_date]
test = df[df["date"] >= split_date]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# 5. Preprocess data
preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# 6. Build model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# 7. Train and evaluate
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["No stock-out", "Stock-out"],
    zero_division=0
))

# 8. Save trained model
joblib.dump(
    {"model": model, "features": features},
    "stockout_model.joblib"
)

print("\nSaved stockout_model.joblib")
