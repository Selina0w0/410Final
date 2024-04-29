#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
import sparktorch
import os
import requests
import torch
import subprocess


# In[2]:


from sparktorch import SparkTorch, create_spark_torch_model, serialize_torch_obj, PysparkPipelineWrapper
import torch.nn as nn
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from cnn_network import Net


# In[8]:


spark = SparkSession.builder.appName("Final").getOrCreate()


# In[9]:


script_dir = os.path.dirname(os.path.abspath("skf5373.ipynb"))
dataset_url = "https://raw.githubusercontent.com/dmmiller612/sparktorch/master/examples/mnist_train.csv"
dataset_path = os.path.join(script_dir, "mnist_train.csv")

if not os.path.exists(dataset_path):
    subprocess.run(['wget', dataset_url, '-O', dataset_path])


# In[10]:


df = spark.read.option("inferSchema", "true").csv(dataset_path).orderBy(rand()).repartition(2)
network = Net()


# In[11]:


# Build the pytorch object
torch_obj = serialize_torch_obj(
    model=network,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    lr=0.001
)


# In[12]:


# Setup features
vector_assembler = VectorAssembler(inputCols=df.columns[1:785], outputCol='features')


# In[13]:


spark_model = SparkTorch(
    inputCol='features',
    labelCol='_c0',
    predictionCol='predictions',
    torchObj=torch_obj,
    iters=50,
    verbose=1,
    validationPct=0.2,
    miniBatch=128
)


# In[ ]:


# Create and save the Pipeline
p = Pipeline(stages=[vector_assembler, spark_model]).fit(df)
p.write().overwrite().save('cnn')


# In[ ]:


# Example of loading the pipeline
loaded_pipeline = PysparkPipelineWrapper.unwrap(PipelineModel.load('cnn'))


# In[ ]:


# Run predictions and evaluation
predictions = loaded_pipeline.transform(df).persist()

evaluator = MulticlassClassificationEvaluator(
    labelCol="_c0", predictionCol="predictions", metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
print("Train accuracy = %g" % accuracy)


# In[78]:


spark.stop()


# In[ ]:




