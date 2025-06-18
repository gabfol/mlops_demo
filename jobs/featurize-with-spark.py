from pyspark.sql import SparkSession
import pandas as pd

# Start Spark session
spark = SparkSession.builder \
    .appName("WineQualityPrep") \
    .getOrCreate()


# Load data using pandas then convert to spark dataframe 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
df.head()

#convert to spark dataframe... just for demo
#df_spark = spark.createDataFrame(df)
df_spark = spark.createDataFrame(df.values.tolist(), schema=df.columns.tolist())


# Rename columns: remove spaces
for col_name in df_spark.columns:
    df_spark = df_spark.withColumnRenamed(col_name, col_name.replace(" ", "_"))

# Preview
df_spark.printSchema()
df_spark.show(5)


# Cast "quality" column to integer if not already
df_spark = df_spark.withColumn("quality", df_spark["quality"].cast("int"))

# Show a few rowsw
df_spark.show(5)

# Optionally: normalize features (example: min-max scaling one column)
from pyspark.sql.functions import col, min, max

feature = "alcohol"
min_val = df_spark.agg({feature: "min"}).collect()[0][0]
max_val = df_spark.agg({feature: "max"}).collect()[0][0]

df_spark = df_spark.withColumn(
    f"{feature}_scaled",
    (col(feature) - min_val) / (max_val - min_val)
)

df_spark.select("alcohol", "alcohol_scaled").show(5)

spark.stop()  # close Spark session explicitly