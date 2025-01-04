import json
import base64
import requests

class LLaVAClient:
    def __init__(self, api_base = 'http://localhost:8001'):
        self.api_base = api_base + '/generate'
        self.template = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT:"""

    def preprocess(self, image):
        """
        image: path of image
        """
        with open(image, 'rb') as f:
            image_file = f.read()
            encoded = base64.b64encode(image_file).decode('utf-8')
        return encoded

    def run(self, encoded_image, prompt):
        """
        encoded_image: image base64
        prompt: str
        """
        data = {
            "prompt": self.template.format(prompt = prompt),
            'max_tokens': 512,
            'temperature': 0.01,
            'stop': '<|endoftext|>',
            'top_p': 0.7,
            'images': [encoded_image],  # str or a list of str. can be **url** or **base64.**  must match the number of '<image>'
        }

        res = requests.post(self.api_base, json=data)
        print(res.text)
        js = json.loads(res.text)['text'][0].split('ASSISTANT:')[1]
        return js

    def run_text(self, prompt):
        template = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {prompt} ASSISTANT:"""
        data = {
            "prompt": template.format(prompt = prompt),
            'max_tokens': 512,
            'temperature': 0.01,
            'stop': '<|endoftext|>',
            'top_p': 0.7
        }

        res = requests.post(self.api_base, json=data)
        print(res.text)
        js = json.loads(res.text)['text'][0].split('ASSISTANT:')[1]
        return js
    
    def test_post(self, data):
        ret = requests.post(self.api_base, json=data)
        print(ret.json()['text'][0].split('ASSISTANT:')[1].strip())
        
    def test(self, encoded_image, prompt):
        data = {
            "prompt": self.template.format(prompt = prompt),
            'max_tokens': 512,
            'temperature': 0.01,
            'stop': '<|endoftext|>',
            'top_p': 0.7,
            'images': [encoded_image],  # str or a list of str. can be **url** or **base64.**  must match the number of '<image>'
        }
        
        import time
        from multiprocessing import pool
        P = 16
        N = 256
        assert(N / P == N // P), "total tasks should be exactly divided by process number"
        datas = [[data for _ in range(N // P)] for _ in range(P)] # 平均分配任务
        p = pool.Pool(P)
        
        st = time.time()
        p.map(self.test_post, datas)
        p.close()
        p.join()
        et = time.time()
        spent = et - st
        print("Processing {} data spent time: {:.2f} s, qps: {:.2f}".format(N, spent, N / spent))
        

if __name__ == "__main__":
    client = LLaVAClient()
    # warm up
    prompt = "Hello, can you help me?"
    ret = client.run_text(prompt)
    print(ret)

    # test
    image = "./images/test.jpg"
    prompt = "Describe this picture"
    encoded_image = client.preprocess(image)
    client.test(encoded_image, prompt)
    
# ----------------
# There also a offline batched mode 
#from vllm import LLM, SamplingParams, LLaVA
#llm = LLaVA(model="...",  gpu_memory_utilization=0.95) 
#prompts = [ 'prompt1',
#'prompt2 <image> \n say',
#'prompt3 <image> say something <image> something',
#]
#sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=256)
#outputs = llm.generate(prompts, sampling_params, images=[image]*3) # PIL url or base64。
