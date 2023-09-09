FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7:16
RUN python -m pip install --upgrade pip
RUN pip install \
  --extra-index-url https://download.pytorch.org/whl/cu116 \
  azureml-defaults \
  pandas \
  inference-schema \
  onnxruntime-gpu \
  mltable \
  applicationinsights \
  opencensus-ext-azure