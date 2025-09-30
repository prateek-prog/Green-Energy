import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# ✅ Load dataset
df = pd.read_csv("data/carbon_footprint.csv")

# ✅ Prepare features and target
X = pd.get_dummies(df.drop("CarbonEmission", axis=1))
y = df["CarbonEmission"]

# ✅ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# ✅ Create 'models/' folder safely
model_dir = os.path.join(os.getcwd(), "models")

try:
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"✅ Created directory: {model_dir}")
    else:
        print(f"✅ Directory already exists: {model_dir}")
except Exception as e:
    print(f"❌ Failed to create model directory: {e}")
    exit(1)  # Stop if we can't create folder

# ✅ Define file paths
model_path = os.path.join(model_dir, "mlp.joblib")
scaler_path = os.path.join(model_dir, "scale.joblib")
columns_path = os.path.join(model_dir, "column.joblib")

# ✅ Save model
try:
    with open(model_path, "wb") as f_model:
        joblib.dump(model, f_model)
    print("✅ Model saved successfully.")
except Exception as e:
    print(f"❌ Failed to save model: {e}")

# ✅ Save scaler
try:
    with open(scaler_path, "wb") as f_scaler:
       joblib.dump(scaler, f_scaler)
    print("✅ Scaler saved successfully.")
except Exception as e:
    print(f"❌ Failed to save scaler: {e}")

 #✅ Save column names
try:
    with open(columns_path, "wb") as f_columns:
       joblib.dump(X.columns.tolist(), f_columns)
    print("✅ Columns saved successfully.")
except Exception as e:
    print(f"❌ Failed to save columns: {e}")

print("\n🎉 Model training complete.")
