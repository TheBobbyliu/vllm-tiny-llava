# setting up multiple processes is better than changing the tensor parallel size
# because there is a bottleneck in vision model

mkdir -p execute_log

model_path="path-to-tiny-llava-3.1B-phi"

CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.llava_server \
--model $model_path \
--tensor-parallel-size 1 --gpu-memory-utilization 0.8 --trust-remote-code --vision-tower siglip --port 8001 > execute_log/8001.log &

CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.llava_server \
--model $model_path \
--tensor-parallel-size 1 --gpu-memory-utilization 0.8 --trust-remote-code --vision-tower siglip --port 8002 > execute_log/8002.log &

CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.llava_server \
--model $model_path \
--tensor-parallel-size 1 --gpu-memory-utilization 0.8 --trust-remote-code --vision-tower siglip --port 8003 > execute_log/8003.log &

CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.llava_server \
--model $model_path \
--tensor-parallel-size 1 --gpu-memory-utilization 0.8 --trust-remote-code --vision-tower siglip --port 8004 > execute_log/8004.log &

# on nginx, configure as follows to forward the requests
"""
upstream vllm_app {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
    server 127.0.0.1:8004;
}

server {
  listen 8000;

  location /generate {
    proxy_pass http://vllm_app/generate;
  }
}
"""
# steps:
# 1. install nginx: yum/apt-get install nginx
#    execute `service nginx start`
# 2. create your own nginx configuration file:
#    touch /etc/nginx/conf.d/vllm_llava.conf
#    paste configurations in the file
# 3. execute `service nginx reload`