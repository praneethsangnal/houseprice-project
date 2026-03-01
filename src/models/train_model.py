import pandas as pd
import joblib
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.preprocessing.preprocess import preprocess

def train_model():
    BASE_DIR=Path(__file__).resolve().parent.parent.parent
    df=pd.read_csv(BASE_DIR/"datasets"/"dataset.csv")
    modelpath=BASE_DIR/"savedmodels"

    X,y=preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    models={
        "LinearRegression":LinearRegression(),
        "Ridge":Ridge(alpha=0.1),
        "Lasso":Lasso(alpha=0.001,max_iter=10000),
        "RandomForest":RandomForestRegressor(n_estimators=100,random_state=42)

    }

    best_model=None
    best_r2=-1
    best_modelname=""
    y_predbest=None
    results={}

    for modelname,model in models.items():
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        x=r2_score(y_test,y_pred)
        results[modelname]=x

        if(x>best_r2):
            best_r2=x
            best_model=model
            best_modelname=modelname
            y_predbest=y_pred
    
    print("\nModel Comparison")
    print("------------------")
    for name, score in results.items():
        print(f"{name:} {score:.4f}")
    
    print("best model is",best_modelname)
    print("r2 score is",best_r2)
    print("mae score is",mean_absolute_error(y_test,y_predbest))
    print("rmse score is",mean_squared_error(y_test,y_predbest)**0.5)

    joblib.dump(best_model,modelpath/"model.pkl")
    joblib.dump(scaler,modelpath/"scaler.pkl")
    joblib.dump(results,modelpath/"model_scores.pkl")
    joblib.dump(best_modelname,modelpath/"best_modelname.pkl")


if(__name__=="__main__"):
    train_model()



    