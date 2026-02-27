import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

    model=LinearRegression()
    model.fit(X_train,y_train)

    joblib.dump(model,modelpath/"linear_model.pkl")
    joblib.dump(scaler,modelpath/"scaler.pkl")

    y_pred=model.predict(X_test)

    print("MAE",mean_absolute_error(y_test,y_pred))
    print("RMSE",mean_squared_error(y_test,y_pred)**0.5)
    print("R2",r2_score(y_test,y_pred))

if(__name__=="__main__"):
    train_model()



    