FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7:16
RUN python -m pip install --upgrade pip
RUN pip install \
  pandas \
  inference-schema \
  onnxruntime-gpu \
  mltable \
  applicationinsights \
  opencensus-ext-azure \
  azureml-defaults \
  azureml-ai-monitoring~=0.1.0b1