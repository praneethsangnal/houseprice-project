import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ----------------------------
# Load Paths & Artifacts
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "savedmodels"

model = joblib.load(MODEL_DIR / "model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
medians = joblib.load(MODEL_DIR / "medians.pkl")
categories = joblib.load(MODEL_DIR / "categories.pkl")
columns = joblib.load(MODEL_DIR / "columns.pkl")


# ----------------------------
# Prediction Function
# ----------------------------
def predict(input_dict):

    df = pd.DataFrame([input_dict])

    # ---------- Numeric Handling ----------
    for col, median in medians.items():

        if col not in df.columns:
            df[col] = median

        df[col] = pd.to_numeric(df[col], errors="coerce")

        if pd.isna(df[col].iloc[0]):
            df[col] = median

    # ---------- Categorical Handling ----------
    for col, allowed in categories.items():

        if col not in df.columns:
            df[col] = "None"

        elif df[col].iloc[0] not in allowed:
            df[col] = "None"

    # ---------- Encoding ----------
    df = pd.get_dummies(df, drop_first=True)

    # ---------- Feature Alignment ----------
    df = df.reindex(columns=columns, fill_value=0)

    # ---------- Scaling ----------
    df_scaled = scaler.transform(df)

    # ---------- Prediction ----------
    pred_log = model.predict(df_scaled)[0]
    pred_price = np.expm1(pred_log)

    return pred_log, pred_price


# ----------------------------
# Run Example
# ----------------------------
if __name__ == "__main__":

    sample_input = {
        "GrLivArea": 2000,
        "Neighborhood": "CollgCr",
        "TotalBsmtSF": 1000,
        "YearBuilt": 2005,
        "LotArea": None
    }

    log_price, actual_price = predict(sample_input)

    #ADD THIS HERE
    model_name = joblib.load(MODEL_DIR / "best_modelname.pkl")

    print("\n" + "=" * 40)
    print("🏠 HOUSE PRICE PREDICTION")
    print("=" * 40)

    print(f"\n🤖 Model Used: {model_name}")   # 👈 HERE

    print("\n📥 Input:")
    for k, v in sample_input.items():
        print(f"{k:15}: {v}")

    print("\n📊 Prediction:")
    print(f"Log Price       : {log_price:.4f}")
    print(f"Estimated Price : ${actual_price:,.2f}")

    print("\n" + "=" * 40)