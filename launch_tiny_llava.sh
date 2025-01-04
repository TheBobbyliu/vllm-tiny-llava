CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -m \
vllm.entrypoints.llava_server \
--model path-to-tiny-llava-3.1B-phi \
--tensor-parallel-size 1 \
--gpu-memory-utilization 0.8 \
--trust-remote-code \
--vision-tower siglip --port 8000
