import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)



st.set_page_config(
    page_title="Random Forest Classifier",
    layout="wide"
)

st.title("Random Forest Classification with Breast Cancer Dataset")



st.sidebar.title("Model Settings")

missing_strategy = st.sidebar.selectbox(
    "Missing Value Strategy",
    ["Mean", "Median", "Most Frequent", "Drop Rows"]
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.5,
    value=0.25,
    step=0.05
)

n_estimators = st.sidebar.selectbox(
    "Number of Trees",
    [50, 100, 150, 200]
)

max_depth = st.sidebar.selectbox(
    "Max Depth",
    [3, 5, 7, 10, None]
)

min_samples_split = st.sidebar.selectbox(
    "Min Samples Split",
    [2, 5, 10]
)



st.header("1. Data Ingestion")

@st.cache_data
def load_data():

    data = load_breast_cancer(as_frame=True)

    df = data.frame

    raw_path = os.path.join(
        RAW_DIR,
        "breast_cancer_dataset.csv"
    )

    df.to_csv(raw_path, index=False)

    # Add Missing Values

    np.random.seed(42)

    for col in df.columns[:-1]:

        df.loc[
            df.sample(frac=0.1).index,
            col
        ] = np.nan

    return df

df = load_data()

st.success("Breast Cancer Dataset Loaded Successfully")

st.dataframe(df, use_container_width=True)



st.header("2. Data Cleaning")

df_clean = df.copy()

if missing_strategy == "Drop Rows":

    df_clean = df_clean.dropna()

else:

    fill_map = {
        "Mean": "mean",
        "Median": "median",
        "Most Frequent": "most_frequent"
    }

    imputer = SimpleImputer(
        strategy=fill_map[missing_strategy]
    )

    cols = df_clean.select_dtypes(include=np.number).columns

    df_clean[cols] = imputer.fit_transform(
        df_clean[cols]
    )

st.dataframe(df_clean, use_container_width=True)



if st.button("Save Cleaned Dataset"):

    clean_path = os.path.join(
        CLEAN_DIR,
        "cleaned_breast_cancer_dataset.csv"
    )

    df_clean.to_csv(clean_path, index=False)

    st.success("Cleaned Dataset Saved Successfully")



st.header("3. Load Cleaned Dataset")

files = [
    f for f in os.listdir(CLEAN_DIR)
    if "breast_cancer_dataset" in f
]

if not files:

    st.warning("No cleaned dataset found")

    st.stop()

selected_file = st.selectbox(
    "Select Dataset",
    files
)

data = pd.read_csv(
    os.path.join(CLEAN_DIR, selected_file)
)

st.dataframe(data, use_container_width=True)



st.header("4. Model Training")

X = data.drop(columns=["target"])

y = data["target"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



param_grid = {
    "n_estimators": [n_estimators],
    "max_depth": [max_depth],
    "min_samples_split": [min_samples_split]
}



grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

# Predictions

pred = model.predict(X_test)



accuracy = accuracy_score(y_test, pred)

st.success(f"Accuracy Score: {accuracy:.4f}")

st.subheader("Best Parameters")

st.write(grid.best_params_)

st.subheader("Classification Report")

st.text(classification_report(y_test, pred))



st.header("5. Confusion Matrix")

cm = confusion_matrix(y_test, pred)

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

ax.set_xlabel("Predicted")

ax.set_ylabel("Actual")

ax.set_title("Confusion Matrix")

st.pyplot(fig)



st.header("6. Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig2, ax2 = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature",
    palette="viridis",
    ax=ax2
)

ax2.set_title("Top 10 Important Features")

st.pyplot(fig2)



st.header("7. Actual vs Predicted")

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": pred
})

st.dataframe(
    results.head(20),
    use_container_width=True
)



st.header("8. Save Model")

model_path = os.path.join(
    MODEL_DIR,
    "random_forest_classifier.pkl"
)

with open(model_path, "wb") as f:

    pickle.dump(model, f)

st.success(f"Model Saved Successfully At:\n{model_path}")


with open(model_path, "rb") as file:

    st.download_button(
        label="Download Trained Model",
        data=file,
        file_name="random_forest_classifier.pkl",
        mime="application/octet-stream"
    )



st.header("9. Sample Predictions")

sample = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": pred[:10]
})

st.dataframe(sample, use_container_width=True)