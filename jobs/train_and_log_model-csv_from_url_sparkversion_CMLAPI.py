## https://docs.cloudera.com/machine-learning/cloud/api/topics/ml-api-v2.html

# !pip3 install https://$CDSW_DOMAIN/api/v2/python.tar.gz

import cmlapi
import os 

from pyspark.sql import SparkSession
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd


#cml_domain= "https://emerging-cml.emerging.dp5i-5vkq.cloudera.site"
cml_domain = os.environ.get("CML_DOMAIN")

# cml_gab_api_key= "1cb1ac9ae90ad712362ab1864fedacd7a4605238da10def85ba29b5aba8c96e1.5b7b3076d152183f78acf998e6cad09b579f1024ae4e95538e15e87789fe65dc"
cml_gab_api_key = os.environ.get("CML_GAB_API_KEY")

# connect to CML API
api_client=cmlapi.default_client(url=cml_domain,cml_api_key=cml_gab_api_key)
# test if connection successful
api_client.list_projects()

# set additional params required to remotely log experiments runs
cml_project_id = os.environ.get("CDSW_PROJECT_ID")
cml_mlflow_experiment_id = os.environ.get("WINE_QUALITY_EXPERIMENT_ID")

# crea il body per lanciare un experimentrun
body = cmlapi.ExperimentRunRequest(
    project_id = cml_project_id,
    experiment_id = cml_mlflow_experiment_id,
    run_name = "elasticnet_wine-job-using-cmlapi"
)

api_response = api_client.create_experiment_run(body)


# initialize spark driver 
spark = SparkSession.builder.appName("mlflow-elasticnet-demo").getOrCreate()

# Load data using pandas then convert to spark dataframe 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

#convert to spark dataframe... just for demo
#df_spark = spark.createDataFrame(df)
df_spark = spark.createDataFrame(df.values.tolist(), schema=df.columns.tolist())

# reconvert to pands, to keep same code below (just a demo)
df = df_spark.toPandas()  # if you still want to use scikit-learn

# Display data
df.head()


# Extract features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Set up experiment and start run

# DA MODIFICARE USANDO LE CMLAPI: 
mlflow.set_experiment("wine_quality_experiment")

with mlflow.start_run(run_name="elasticnet_wine-job1008"):

    alpha = 0.5
    l1_ratio = 0.5
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    mlflow.set_tag("Training Info", "Basic model for wine quality - from Spark job")

    signature = infer_signature(X_train, model.predict(X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="elasticnet_model",
        signature=signature,
        input_example=X_test,
        registered_model_name="ElasticNetWineModel"
    )
