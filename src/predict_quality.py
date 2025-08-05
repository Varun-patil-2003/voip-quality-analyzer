import joblib
import pandas as pd
from extract_features import extract_features

model = joblib.load("models/mos_predictor.pkl")

def predict_mos(file_path):
    features = extract_features(file_path)
    df = pd.DataFrame([features])
    mos = model.predict(df)[0]
    return round(mos, 2)

if __name__ == "__main__":
    mos_score = predict_mos("data/temp.wav")
    print(f"Predicted MOS: {mos_score}")
