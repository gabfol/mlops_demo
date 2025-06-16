import boto3
import os
import pandas as pd
from io import StringIO
from datetime import datetime, timezone, timedelta

import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from mlflow.models import infer_signature


# 3. Load data: 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Display data
df.head()


# 4. Split data
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test.head()

# 5. Set up experiment and start run
mlflow.set_experiment("wine_quality_experiment")

# override sys.argv with arguments conveyed via JOB_ARGUMENTS variable:
import sys, os, shlex
sys.argv = ["script"] + shlex.split(os.environ.get("JOB_ARGUMENTS", ""))
print(sys.argv)

with mlflow.start_run(run_name="elasticnet_wine-job"):

    # Log hyperparameters
    alpha = 0.5
    l1_ratio = 0.5
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)

    # Train model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate and log metrics
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)


    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic model for wine quality - from job")

    # Infer the model signature
    signature = infer_signature(X_train, model.predict(X_train))
    
    # 6. Log and register model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="elasticnet_model",
        signature=signature,
        input_example=X_test,
        registered_model_name="ElasticNetWineModel"
    )