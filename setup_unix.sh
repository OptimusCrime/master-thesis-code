#!/bin/bash

# Get correct pip
THIS_PIP="$(which pip)"

# Install/upgrade requirements from requirements file
$THIS_PIP install --upgrade -r requirements.txt

# Install/upgrade tensorflow
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    $THIS_PIP install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp34-cp34m-linux_x86_64.whl
elif [[ "$OSTYPE" == "darwin"* ]]; then
    $THIS_PIP install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl
fi

# Install graphviz (used for visualization of Keras)
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    sudo apt-get install graphviz
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install graphviz
fi
