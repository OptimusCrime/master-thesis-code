#!/bin/bash

# Get correct pip
THIS_PIP="$(which pip)"

# Install/upgrade requirements from requirements file
$THIS_PIP install --upgrade -r requirements.txt

# Install/upgrade tensorflow
$THIS_PIP install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py3-none-any.whl
