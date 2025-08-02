HF_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
# HF_MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
# HF_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
# HF_MODEL_PATH="mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_USERNAME="$1"
HF_TOKEN="$2"


# git config --global credential.helper store
# huggingface-cli login --token $HF_TOKEN --add-to-git-credential


# Install git-lfs if needed
sudo apt-get update -y && sudo apt-get install git-lfs -y --no-install-recommends
git lfs install

# Clone the Hugging Face model repository
HF_MODEL_DIR=~/tensorrt/hf_model
[ -d $HF_MODEL_DIR ] && rm -rf $HF_MODEL_DIR
mkdir -p $HF_MODEL_DIR && cd $HF_MODEL_DIR
GIT_ASKPASS=echo git clone https://$HF_USERNAME:$HF_TOKEN@huggingface.co/$HF_MODEL_PATH .
