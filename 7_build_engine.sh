#!/bin/bash
# Description: This script automates the installation and inference process for a Hugging Face model using TensorRT-LLM. Ensure that Git and Git LFS ('apt-get install git-lfs') are installed before running this script. Before running this script, run the following scripts sequentially: 1. install_git_and_lfs.sh 2. install_tensorrt_llm.sh
EXAMPLE_MODEL_NAME=llama
HF_MODEL_DIR=/tensorrt/hf_model
QUANT_MODEL_DIR=/tensorrt/quant_hf_model
CHECKPOINT_DIR=/tensorrt/trt-checkpoints/fp16/1-gpu
ENGINE_DIR=/tensorrt/trt-engines/fp16/1-gpu

# python3 /tensorrt/tensorrtllm_backend/tensorrt_llm/examples/quantization/quantize.py --model_dir $HF_MODEL_DIR --qformat fp8 --kv_cache_dtype fp8 --output_dir $QUANT_MODEL_DIR

# Convert the model checkpoint to TensorRT format
echo "--------> Converting Model ------------"
# pip install -r /tensorrt/tensorrtllm_backend/tensorrt_llm/examples/$EXAMPLE_MODEL_NAME/requirements.txt
python /tensorrt/tensorrtllm_backend/tensorrt_llm/examples/$EXAMPLE_MODEL_NAME/convert_checkpoint.py \
    --model_dir $HF_MODEL_DIR \
    --output_dir $CHECKPOINT_DIR \
    --dtype bfloat16 \
    --tp_size 1 \
    --workers 4
    # --dtype bfloat16
    # --model_dir $QUANT_MODEL_DIR \

# No need to specify dtype, it will automatically be inferred from "torch_dtype" value in config.json of the model

# Build TensorRT engine
echo "--------> Building engine ------------" 
trtllm-build --checkpoint_dir $CHECKPOINT_DIR \
    --output_dir $ENGINE_DIR \
    --bert_attention_plugin auto \
    --gpt_attention_plugin auto \
    --context_fmha enable \
    --remove_input_padding enable \
    --use_paged_context_fmha enable \
    --use_fused_mlp enable \
    --kv_cache_type paged \
    --paged_state enable \
    --gemm_plugin bfloat16 \
    --max_batch_size 1024 \
    --workers 4 \
    --max_input_len 4096 \
    --max_seq_len 4352
    # --multiple_profiles enable \

# --multiple_profiles enable
# Enables multiple TensorRT optimization profiles in the built engines,
# will benefits the performance especially when GEMM plugin is disabled,
# because more optimization profiles help TensorRT have more chances to select better kernels.
# Note: This feature increases engine build time but no other adverse effects are expected.

# Set --gemm_plugin auto if --multiple_profiles is not enabled


# Run inference with the TensorRT engine
# echo "--------> Running Inference ------------" 
# python3 /tensorrt/tensorrtllm_backend/tensorrt_llm/examples/run.py \
#     --max_output_len=250 \
#     --tokenizer_dir $HF_MODEL_DIR \
#     --engine_dir=$ENGINE_DIR \
#     --max_attention_window_size=4096 \
#     --temperature=0.3 \
#     --top_k=50 \
#     --top_p=0.9 \
#     --repetition_penalty=1.2 \
#     --input_text="what is machine learning?"
#     # --tokenizer_dir $QUANT_MODEL_DIR \
