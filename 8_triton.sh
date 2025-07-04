[ -d "/tensorrt/triton-models/" ] && rm -rf "/tensorrt/triton-models/"
mkdir -p  /tensorrt/triton-models/
cp /tensorrt/tensorrtllm_backend/all_models/inflight_batcher_llm/* /tensorrt/triton-models/ -r


ENGINE_DIR=/tensorrt/trt-engines/fp16/1-gpu
TOKENIZER_DIR=/tensorrt/hf_model
MODEL_FOLDER=/tensorrt/triton-models
TRITON_MAX_BATCH_SIZE=128
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=0
MAX_QUEUE_SIZE=0
FILL_TEMPLATE_SCRIPT=/tensorrt/tensorrtllm_backend/tools/fill_template.py
DECOUPLED_MODE=true # must be true for streaming
LOGITS_DATATYPE=TYPE_FP32
EXCLUDE_INPUT_IN_OUTPUT=true
# KV_CACHE_FREE_GPU_MEM_FRAC=0.5

python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},logits_datatype:${LOGITS_DATATYPE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
# python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE},encoder_input_features_data_type:TYPE_FP16,logits_datatype:${LOGITS_DATATYPE},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRAC}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE},encoder_input_features_data_type:TYPE_FP16,logits_datatype:${LOGITS_DATATYPE},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},logits_datatype:${LOGITS_DATATYPE}


python3 /tensorrt/tensorrtllm_backend/scripts/launch_triton_server.py --world_size=4 --model_repo=${MODEL_FOLDER}


# curl -X POST localhost:8000/v2/models/ensemble/generate_stream -d '{"text_input": "What exactly is a Large Language Model?", "max_tokens": 1000, "stream": true, "bad_words": "", "stop_words": "", "accumulate_tokens": true}'
# curl -X POST localhost:8000/v2/models/ensemble/generate_stream -d '{"text_input": "What exactly is differentiation?", "max_tokens": 1000, "stream": true, "bad_words": "", "stop_words": "", "accumulate_tokens": true}'


# curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What exactly is a Large Language Model? explain in great detail.", "max_tokens": 4000, "bad_words": "", "stop_words": ""}' -o output3.txt
# {"model_name":"ensemble","model_version":"1","sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"\n\nLarge language models (LLMs) are artificial intelligence (AI) models that are trained on vast amounts of text data to generate language outputs that are coherent and natural-sounding. These models have become increasingly popular in recent years due to their ability to generate text that is often indistinguishable from human-written text. In this answer, we will explore what LLMs are, how they work, and some of the applications they have.\n\nWhat is a Large Language Model?\n\nA large language model is a type of AI model that is trained on a large corpus of text data, such as books, articles, or websites. The goal of training an LLM is to enable the model to generate language outputs that are coherent and natural-sounding, and that can be used for a variety of applications such as language translation, text summarization, and chatbots.\n\nHow does a Large Language Model Work?\n\nLLMs work by using a technique called deep learning, which involves training multiple layers of artificial neural networks on large amounts of text data. The neural networks in an LLM are designed to learn the patterns and structures of language, such as grammar, syntax, and semantics, by analyzing the training data.\n\nThe training process for an LLM typically involves feeding the model a large corpus of text data and adjusting the model's parameters to minimize the error between the model's output and the correct output. This process is repeated many times until the model is able to generate language outputs that are accurate and coherent.\n\nOnce an LLM is trained, it can be used for a variety of applications such as language translation, text summarization, and chatbots. For example, an LLM can be trained to translate text from one language to another, or to summarize long documents into shorter summaries.\n\n"}

# curl -X POST localhost:8000/v2/models/tensorrt_llm_bls/generate -d '{"text_input": "What exactly is a Large Language Model?", "max_tokens": 400, "bad_words": "", "stop_words": ""}'
# {"model_name":"tensorrt_llm_bls","model_version":"1","text_output":"\n\nLarge language models (LLMs) are artificial intelligence (AI) models that are trained on vast amounts of text data to generate language outputs that are coherent and natural-sounding. These models have become increasingly popular in recent years due to their ability to generate text that is often indistinguishable from human-written text. In this answer, we will explore what LLMs are, how they work, and some of the applications they have.\n\nWhat is a Large Language Model?\n\nA large language model is a type of AI model that is trained on a large corpus of text data, such as books, articles, or websites. The goal of training an LLM is to enable the model to generate language outputs that are coherent and natural-sounding, and that can be used for a variety of applications such as language translation, text summarization, and chatbots.\n\nHow does a Large Language Model Work?\n\nLLMs work by using a technique called deep learning, which involves training multiple layers of artificial neural networks on large amounts of text data. The neural networks in an LLM are designed to learn the patterns and structures of language, such as grammar, syntax, and semantics, by analyzing the training data.\n\nThe training process for an LLM typically involves feeding the model a large corpus of text data and adjusting the model's parameters to minimize the error between the model's output and the correct output. This process is repeated many times until the model is able to generate language outputs that are accurate and coherent.\n\nOnce an LLM is trained, it can be used for a variety of applications such as language translation, text summarization, and chatbots. For example, an LLM can be trained to translate text from one language to another, or to summarize long documents into shorter summaries.\n\n"}
