import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")  # local MLflow server
mlflow.set_experiment("housing-price-prediction")

# Start MLflow run
with mlflow.start_run() as run:
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    # Infer model signature and input example
    signature = infer_signature(X_test, preds)
    input_example = X_test.iloc[:2]

    # Log metric
    mlflow.log_metric("rmse", rmse)

    # Log model with metadata
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    # Register the model
    result = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name="housing-price-model"
    )

    print(f" Model registered as: {result.name}, version: {result.version}")
    print(f" RMSE: {rmse:.4f}")
    print(f" View run: http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
