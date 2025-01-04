python -m vllm.entrypoints.llava_server \
--model path-to-llava-1.5-7b-hf \
--tensor-parallel-size 1 \
--gpu-memory-utilization 0.9 \
--dtype half --port 8000

