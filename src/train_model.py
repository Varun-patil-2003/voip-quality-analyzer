import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/features_mos.csv")

X = df.drop(columns=["mos"])
y = df["mos"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "models/mos_predictor.pkl")
print("Model trained and saved.")
