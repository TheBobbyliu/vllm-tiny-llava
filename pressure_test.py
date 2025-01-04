import os
import time
from glob import glob
from multiprocessing import Process
from llava_client import LLaVAClient
from tqdm import tqdm
num_thread = 8
task_cnt = 100
inp = "images/test.jpg"
prompt = "describe this picture"

def process(cnt = 100):
    client = LLaVAClient()
    encoded = client.preprocess(inp)
    for i in tqdm(range(cnt)):
        response = client.run(encoded, prompt)

start = time.time()
ps = [Process(target = process, args = (task_cnt, )) for _ in range(num_thread)]
[p.start() for p in ps]
[p.join() for p in ps]
end = time.time()
print("Processing", num_thread * task_cnt , "tasks, spent:", end - start, 's, qps:',  num_thread * task_cnt / (end - start))

