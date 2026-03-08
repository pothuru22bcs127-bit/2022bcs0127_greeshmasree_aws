import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("data/housing.csv")

# Features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# One-hot encoding for categorical columns
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, pred, squared=False)
r2 = r2_score(y_test, pred)

print("RMSE:", rmse)
print("R2:", r2)

# Save model
joblib.dump(model, "app/model.pkl")
