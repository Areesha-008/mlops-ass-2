# src/train.py

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

        # Log params
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


if __name__ == "__main__":

    mlflow.set_experiment("mlops-project")

    # Run 1
    train_and_log(
        RandomForestClassifier(n_estimators=50),
        "RandomForest",
        {"n_estimators": 50}
    )

    # Run 2
    train_and_log(
        RandomForestClassifier(n_estimators=100),
        "RandomForest",
        {"n_estimators": 100}
    )

    # Run 3
    train_and_log(
        SVC(kernel="linear"),
        "SVM",
        {"kernel": "linear"}
    )