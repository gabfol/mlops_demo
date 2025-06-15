import sys
from pyspark.sql import SparkSession
 

# Initialize Spark session
#spark = SparkSession.builder.appName("HelloWorldApp").getOrCreate()

spark = SparkSession.builder.appName("HelloWorldApp").config("spark.executor.instances", "2").getOrCreate()

print(f"Driver Python version: {sys.version}")

# Create a simple RDD and print its contents
rdd = spark.sparkContext.parallelize(["Hello", "World", "from", "Spark", "!"])
for word in rdd.collect():
    print(word)

spark.stop()