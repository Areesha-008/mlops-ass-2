# src/train.py

import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data_loader import load_data
from preprocess import split_data


def train_and_log(model, model_name, params):
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    with mlflow.start_run():

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        mlflow.log_param("model_name", model_name)

        # Log metric
        mlflow.log_metric("accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="IrisClassifier"
        )

        print(f"{model_name} -> Accuracy: {acc}")

        return acc, model


if __name__ == "__main__":

    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("mlops-project")

    best_accuracy = 0
    best_model = None

    # 🔹 Run 1
    acc, model = train_and_log(
        RandomForestClassifier(n_estimators=50),
        "RandomForest",
        {"n_estimators": 50}
    )
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

    # 🔹 Run 2
    acc, model = train_and_log(
        RandomForestClassifier(n_estimators=100),
        "RandomForest",
        {"n_estimators": 100}
    )
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

    # 🔹 Run 3
    acc, model = train_and_log(
        SVC(kernel="linear"),
        "SVM",
        {"kernel": "linear"}
    )
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

    # ✅ Save best model
    joblib.dump(best_model, "models/model.pkl")

    print("\nBest Model Saved!")
    print("Best Accuracy:", best_accuracy)