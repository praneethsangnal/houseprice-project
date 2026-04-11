# 🏠 House Price Prediction using Machine Learning

## 📌 Overview

This project is a **machine learning-based web application** that predicts house prices based on user inputs. It combines **data preprocessing, model comparison, feature importance, and deployment** to build a complete end-to-end intelligent system.

The system is deployed using **Streamlit + Render**, enabling real-time predictions through an interactive UI.

---

## 🌐 Live Demo

👉 Try the app here:
https://houseprice-project-3906.onrender.com

---

## 🚀 Features

* 🏠 House price prediction using Machine Learning
* 📊 Model comparison across multiple algorithms
* 🧠 Feature importance using Random Forest
* ⚙️ Robust preprocessing pipeline (handles missing & unseen data)
* 📈 Derived insights:

  * Price per sq ft
  * Property age
  * Quality score
* 🎨 Clean and structured Streamlit UI
* ⚡ Real-time predictions with user input

---

## 🧠 Model Details

### 🔹 Models Used

* Linear Regression
* Ridge Regression
* Lasso Regression
* Random Forest Regressor

---

### 🔹 Model Comparison

| Model             | R² Score     |
| ----------------- | ------------ |
| Linear Regression | 0.8383       |
| Ridge Regression  | 0.8401       |
| Lasso Regression  | 0.8645       |
| Random Forest     | **0.8852** ✅ |

---

### 🔹 Best Model

👉 **Random Forest Regressor**

---

### 🔹 Why Random Forest?

* Handles non-linear relationships
* Robust to multicollinearity
* Automatically captures feature interactions
* Performs best without heavy feature engineering

---

## 📊 Feature Importance

Using Random Forest, the most important features were:

* ⭐ Overall Quality (`OverallQual`)
* 📐 Living Area (`GrLivArea`)
* 🚗 Garage Capacity (`GarageCars`)
* 🏚 Basement Area (`TotalBsmtSF`)
* 📅 Year Built (`YearBuilt`)

👉 These align with real-world housing valuation factors.

---

## ⚙️ Data Preprocessing

### 🔹 Missing Values

* Numeric → filled using **median**
* Categorical → replaced with `"None"`

Saved for inference:

* `medians.pkl`
* `categories.pkl`

---

### 🔹 Encoding & Alignment

* One-Hot Encoding using `pd.get_dummies`
* Column alignment using:

  * `columns.pkl`

👉 Ensures consistency between training and prediction.

---

### 🔹 Scaling

* Applied **StandardScaler**
* Saved as:

  * `scaler.pkl`

---

## 🔧 Feature Engineering

Explored features like:

* House Age = Year Sold - Year Built
* Total Area = Living Area + Basement Area

📌 Observation:

* Did **not improve performance**
* Tree-based models already capture these relationships

👉 Included as a **learning insight in project**

---

## 📊 Model Evaluation

* **MAE (Mean Absolute Error)** → average error
* **RMSE** → penalizes large errors
* **R² Score** → model performance

Final metrics:

* R² ≈ **0.885**
* MAE ≈ **0.099**
* RMSE ≈ **0.146**

---

## 🌐 Streamlit Web App

### Features:

* Input house details via sidebar
* Real-time prediction
* Displays:

  * 💰 Estimated price
  * 📊 Price per sq ft
  * 📅 Property age
  * ⭐ Quality score

### Run locally:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
houseprice-project/
│
├── datasets/
│   └── dataset.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── savedmodels/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── medians.pkl
│   ├── categories.pkl
│   ├── columns.pkl
│   ├── model_scores.pkl
│   └── best_modelname.pkl
│
├── src/
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict.py
│   │
│   ├── preprocessing/
│   │   └── preprocess.py
│   │
│   └── utils/
│
├── app.py
├── main.py
├── predict.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train Model:

```bash
python -m src.models.train_model
```

### Run Prediction Script:

```bash
python predict.py
```

### Run Web App:

```bash
streamlit run app.py
```

---

## ⚠️ Limitations

* Model trained on historical dataset (Ames Housing)
* Predictions may differ from real-world market prices
* Limited number of input features in UI

---

## 🚀 Future Improvements

* Add SHAP for advanced interpretability
* Hyperparameter tuning for better accuracy
* Add confidence intervals
* Improve UI/UX further
* Integrate real-time datasets

---

## 💡 Key Learnings

* Model comparison is crucial for performance
* Feature importance improves interpretability
* Tree models reduce need for feature engineering
* Handling unseen inputs is critical in real-world ML
* Deployment bridges gap between ML and product

---

## 👨‍💻 Author

**Praneeth Sangnal**

---

## ⭐ Acknowledgements

* Ames Housing Dataset
* Scikit-learn
* Streamlit
* Render (Deployment)
