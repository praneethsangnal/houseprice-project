import pandas as pd
import numpy as np
from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent.parent.parent
df=pd.read_csv(BASE_DIR/"datasets"/"dataset.csv")
def preprocess(df):
    #very basic preprocessing
    #drop very useless col
    #fill numeric cols with mean
    #fill non numeric with none
    #do one hot encoding
    #replace saleprice with log values
    df=df.drop("Id",axis=1)

    X=df.drop("SalePrice",axis=1)
    y=np.log1p(df["SalePrice"])
    
    num_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    cat_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    X[num_cols]=X[num_cols].fillna(X[num_cols].median())

    X[cat_cols]=X[cat_cols].fillna("None")

    X=pd.get_dummies(X,drop_first=True)

    print(X.head())
    print(y.head())

    return X,y
if(__name__=="__main__"):
    preprocess(df)
