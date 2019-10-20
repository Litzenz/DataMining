# initialize spark session
import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('iteration4').getOrCreate()

# import without schema
df_noschema = spark.read.csv('dataset.csv', header=True)

# data format
from pyspark.sql.types import (StructField, StructType,
                               TimestampType, IntegerType, FloatType)
# define data schema (or use inferSchema =True when loading dataframe)
data_schema = [StructField('instant', IntegerType(), True),
               StructField('dteday', TimestampType(), True),
               StructField('season', IntegerType(), True),
               StructField('yr', IntegerType(), True),
               StructField('mnth', IntegerType(), True),
               StructField('hr', IntegerType(), True),
               StructField('holiday', IntegerType(), True),
               StructField('weekday', IntegerType(), True),
               StructField('workingday', IntegerType(), True),
               StructField('weathersit', IntegerType(), True),
               StructField('temp', FloatType(), True),
               StructField('atemp', FloatType(), True),
               StructField('hum', FloatType(), True),
               StructField('windspeed', FloatType(), True),
               StructField('casual', IntegerType(), True),
               StructField('registered', IntegerType(), True),
               StructField('cnt', IntegerType(), True)]

final_struct = StructType(fields = data_schema)
#import with self-defined schema
df_withschema = spark.read.csv('dataset.csv', schema=final_struct, header=True)
# import with inferred schema automatically, only for csv
df = spark.read.csv('dataset.csv', header=True, inferSchema=True)

#for visualization
data = df.toPandas()

# Data Select
df_selected = df.drop('dteday', 'registered', 'casual', 'season', 'weekday')

# Data Clean
# drop feature 'instant'
df_cleaned = df_selected.drop('instant')
# remove rows where 'cnt' is null
df_cleaned = df_cleaned.na.drop(subset='cnt')
# fill null values with mean of values in 'temp', 'atemp', 'hum'
from pyspark.sql.functions import mean
mean_temp = df.select(mean(df.temp)).collect()[0][0]
mean_atemp = df.select(mean(df.atemp)).collect()[0][0]
mean_hum = df.select(mean(df.hum)).collect()[0][0]
mean = {'temp': mean_temp, 'atemp': mean_atemp, 'hum': mean_hum}
df_cleaned = df_cleaned.na.fill(mean)

# Construct Data
from pyspark.ml.feature import OneHotEncoder
# one hot encode: convert numbers into a vector
mnthEncoder = OneHotEncoder(inputCol='mnth', outputCol='mnthVec')
hrEncoder = OneHotEncoder(inputCol='hr', outputCol='hrVec')
weatherEncoder = OneHotEncoder(inputCol='weathersit', outputCol='weatherVec')
from pyspark.ml import Pipeline
pipeline = Pipeline(stages = [mnthEncoder, hrEncoder, weatherEncoder])
df_constructed = pipeline.fit(df_cleaned).transform(df_cleaned)

# Reduce the data
df_reduced = df_constructed.drop('mnth', 'hr', 'weathersit')

# Project the data
# assemble features into a vector for modeling
from pyspark.ml.feature import VectorAssembler
featuresCol = df_reduced.drop('cnt').columns
assembler = VectorAssembler(inputCols = featuresCol, outputCol = 'features')
df_projected = assembler.transform(df_reduced)

#ready for modeling
final_data = df_projected.select('cnt', 'features')

#Model Select
# select the model
from pyspark.ml.regression import (RandomForestRegressor, 
                                   GBTRegressor, 
                                   DecisionTreeRegressor)

# create evaluator
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(labelCol='cnt', predictionCol='prediction',
                                metricName='r2')

#create a sample for model test
sample, x = final_data.randomSplit([0.1, 0.9])

import numpy as np
# fit models, choose parameters and evaluate
# random forest regression model with maxDepth: 3, 6, 9,..., 30
r2_rfr = np.zeros(10)
for i in np.arange(10):
    rfr = RandomForestRegressor(labelCol='cnt', maxDepth=(i+1)*3)
    rfrModel = rfr.fit(sample)
    prediction_rfr = rfrModel.transform(sample)
    r2_rfr[i] = evaluator.evaluate(prediction_rfr)
r2_rfr

# Gradient Boosted Trees model with maxIter: 10, 20, 30,..., 100
r2_gbt = np.zeros(10)
for i in np.arange(10):
    gbt = GBTRegressor(labelCol='cnt', maxIter = (i+1)*10)
    gbtModel = gbt.fit(sample)
    prediction_gbt = gbtModel.transform(sample)
    r2_gbt[i] = evaluator.evaluate(prediction_gbt)
r2_gbt

# Decision Tree Regression model with maxDepth: 3, 6, 9,..., 30
r2_dtr = np.zeros(10)
for i in np.arange(10):
    dtr = DecisionTreeRegressor(labelCol='cnt', maxDepth= (i+1)*3)
    dtrModel = dtr.fit(sample)
    prediction_dtr = dtrModel.transform(sample)
    r2_dtr[i] = evaluator.evaluate(prediction_dtr)
r2_dtr

# split data into train and test
train, test = final_data.randomSplit([0.7, 0.3])

# Modeling
GBT = GBTRegressor(labelCol='cnt', maxIter = 80)
GBTmodel = GBT.fit(train)
prediction_GBT = GBTmodel.transform(test)

DTR = DecisionTreeRegressor(labelCol='cnt', maxDepth=20)
DTRmodel = DTR.fit(train)
prediction_DTR = DTRmodel.transform(test)

RFR = RandomForestRegressor(labelCol='cnt', maxDepth=20)
RFRmodel = RFR.fit(train)
prediction_RFR = RFRmodel.transform(test)

#evaluate model
r2_GBT = evaluator.evaluate(prediction_GBT)
r2_DTR = evaluator.evaluate(prediction_DTR)
r2_RFR = evaluator.evaluate(prediction_RFR)
print('R2 Score of GBT Regression: ', r2_GBT)
print('R2 Score of Decision Tree Regression: ', r2_DTR)
print('R2 Score of Random Forest Regression: ', r2_RFR)