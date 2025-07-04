curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip
unzip awscliv2.zip
sudo ./aws/install

# docker run -it --rm --net host f39008d83caa /bin/bash

# docker cp 7c40ddf656c7:/tensorrt/trt-engines/fp16/1-gpu ~/tensorrt/trt-engines/qwen-qwq32b-awq-a100-256
# aws s3 cp ~/tensorrt/trt-engines/qwen-qwq32b-awq-a100-256/ s3://triton-llm-weights-2025/qwen-qwq32b-awq-a100-256/ --recursive
