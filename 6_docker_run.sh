TRITON_SERVER_VERSION=25.03
IMAGE_NAME=test-image
CONTAINER_NAME=test

docker build -t ${IMAGE_NAME} .

docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}
docker run --name ${CONTAINER_NAME} -d --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v ~/tensorrt/tensorrtllm_backend:/tensorrt/tensorrtllm_backend \
    -v ~/tensorrt/hf_model:/tensorrt/hf_model \
    -v ~/tensorrt/7_build_engine.sh:/7_build_engine.sh \
    -v ~/tensorrt/8_triton.sh:/8_triton.sh \
    ${IMAGE_NAME}


# docker run --name ${CONTAINER_NAME} -d --net host --shm-size=2g \
#     --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
#     -v ~/tensorrt/tensorrtllm_backend:/tensorrt/tensorrtllm_backend \
#     -v ~/tensorrt/hf_model:/tensorrt/hf_model \
#     -v ~/tensorrt/trt-engines/llama3.1-8b-l4-64:/tensorrt/trt-engines/fp16/1-gpu \
#     -v ~/tensorrt/8_triton.sh:/8_triton.sh \
#     ${IMAGE_NAME}

    # Add this below line if engine files are present
    # -v </path/to/engines>:/engines \
