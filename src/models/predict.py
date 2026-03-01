import pandas as pd
import joblib
import numpy as np
from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent.parent.parent
modelpath=BASE_DIR/"savedmodels"
model=joblib.load(modelpath/"linear_model.pkl")
scaler=joblib.load(modelpath/"scaler.pkl")
medians=joblib.load(modelpath/"medians.pkl")
categories=joblib.load(modelpath/"categories.pkl")
columns=joblib.load(modelpath/"columns.pkl")

def predict():
    input_dict={
        "GrLivArea": 2000,
        "Neighborhood": "CollgCr",
        "TotalBsmtSF": 1000,
        "YearBuilt": 2005,
        "LotArea" : None
    }
    df=pd.DataFrame([input_dict])
    for col,median in medians.items():
        if col not in df.columns or pd.isna(df[col][0]): #if not entered, even if entered not a valid number
            df[col]=median

    for col,category in categories.items():
        if col not in df.columns or df[col][0] not in category:
            df[col]="None"

    df=pd.get_dummies(df,drop_first=True)

    df=df.reindex(columns=columns,fill_value=0)

    df=scaler.transform(df)

    y_pred=model.predict(df)

    print("log value",y_pred[0])

    print("actual value in dollar",np.expm1(y_pred)[0])

if(__name__=="__main__"):
    predict()
    
