import pandas as pd
import joblib
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from src.preprocessing.preprocess import preprocess


def train_model():

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    df = pd.read_csv(BASE_DIR / "datasets" / "dataset.csv")
    modelpath = BASE_DIR / "savedmodels"

    # ✅ ensure folder exists
    modelpath.mkdir(exist_ok=True)

    # ----------------------------
    # Preprocessing
    # ----------------------------
    X, y = preprocess(df)

    # ----------------------------
    # Train-Test Split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # ----------------------------
    # Scaling
    # ----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ----------------------------
    # Models
    # ----------------------------
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=0.1),
        "Lasso": Lasso(alpha=0.001, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    best_model = None
    best_r2 = -1
    best_modelname = ""
    y_predbest = None
    results = {}

    # ----------------------------
    # Training Loop
    # ----------------------------
    for modelname, model in models.items():

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        results[modelname] = r2

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_modelname = modelname
            y_predbest = y_pred

    # ----------------------------
    # Results
    # ----------------------------
    print("\nModel Comparison")
    print("------------------")
    for name, score in results.items():
        print(f"{name:20} {score:.4f}")

    print("\nBest Model:", best_modelname)
    print("R2 Score :", best_r2)
    print("MAE      :", mean_absolute_error(y_test, y_predbest))
    print("RMSE     :", mean_squared_error(y_test, y_predbest) ** 0.5)

    # =====================================================
    # 🔥 FEATURE IMPORTANCE (ONLY FOR RANDOM FOREST)
    # =====================================================
    if best_modelname == "RandomForest":

        importances = best_model.feature_importances_
        feature_names = X.columns

        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        print("\nTop 10 Important Features:")
        print(feature_importance_df.head(10))

        # ✅ Save feature importance
        joblib.dump(
            feature_importance_df,
            modelpath / "feature_importance.pkl"
        )

        # ✅ Plot feature importance
        top_features = feature_importance_df.head(10)

        plt.figure()
        plt.barh(top_features["Feature"], top_features["Importance"])
        plt.gca().invert_yaxis()
        plt.title("Top 10 Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()

    # ----------------------------
    # Save Artifacts
    # ----------------------------
    joblib.dump(best_model, modelpath / "model.pkl")
    joblib.dump(scaler, modelpath / "scaler.pkl")
    joblib.dump(results, modelpath / "model_scores.pkl")
    joblib.dump(best_modelname, modelpath / "best_modelname.pkl")


if __name__ == "__main__":
    train_model()