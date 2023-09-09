FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7:16
RUN python -m pip install --upgrade pip
RUN pip install \
  mlflow \
  azureml-mlflow \
  jsonargparse[signatures] \
  pytorch-lightning[extra] \
  ipykernel \
  onnxruntime-gpu \
  inference-schema \
  pandas \
  mltable \
  scikit-learn \
  onnx