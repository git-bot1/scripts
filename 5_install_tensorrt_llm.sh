#!/bin/bash
# Description: This script automates the installation process for TensorRT-LLM. Prior to running this script, ensure that Git and Git LFS ('apt-get install git-lfs') are installed.
# Step 1: Defining folder path and version
echo "----> Cloning tensorrtllm_backend repo"
TENSORRT_BACKEND_LLM_VERSION=v0.18.0

git clone https://github.com/triton-inference-server/tensorrtllm_backend.git  --branch $TENSORRT_BACKEND_LLM_VERSION
# Update the submodules
cd tensorrtllm_backend
git submodule update --init --recursive
