# Introduction
- This is an adaptation of <a href="https://github.com/vllm-project/vllm"><b>vLLM</b></a>, an easy, fast and cheap LLM serving toolkit, for <a href="https://github.com/TinyLLaVA/TinyLLaVA_Factory"><b>TinyLLaVA</b></a>, a small-scale large multimodal model which achieves better overall performance against existing 7B models.
- This repository is created based on vLLM of version: 0.2.6
- Currently this repo only supports model [TinyLLaVA-3.1B](https://huggingface.co/bczhou/TinyLLaVA-3.1B)


# Log
- [2025/01/03] Initial commit.

# Installation
1. Prepare environment:
```
# install CUDA 12.1 from https://developer.nvidia.com/cuda-toolkit-archive
# create conda environment
conda create -n vllm-llava python=3.10
conda activate vllm-llava
```

2. Install requirements:
```
python3 -m pip install -r requirements-llava-1.txt
python3 -m pip install -r requirements-llava-2.txt
```

3. compile vllm:
```
python3 setup.py develop
```

# Run
### Single process (on one GPU)
1. Edit config *model* in *launch_tiny_llava.sh* to the path of the model
2. Run service with command `sh launch_tiny_llava.sh`, the service will be running on port `8000`
3. Test model with command `python3 tiny_llava_client.py`
### Multiple processes (on multiple GPUs)
1. Edit config *model_path* in *launch_tiny_llava-multiple.sh* to the path of the model, edit script to choose which GPUs to run model on
2. Run multiple services with command `sh launch_tiny_llava-multiple.sh`, services will be running on multiple ports such as `8001, 8002, ...`.
3. Install and configure nginx, to forward requests from port `8000` to other ports. Example can be seen in the script.
4. Test model with command `python3 tiny_llava_client-mp.py`

## Citation

vLLM [paper](https://arxiv.org/abs/2309.06180):
```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```
Tiny-LLaVA [paper](https://arxiv.org/abs/2402.14289):
```BibTeX
@misc{zhou2024tinyllava,
      title={TinyLLaVA: A Framework of Small-scale Large Multimodal Models}, 
      author={Baichuan Zhou and Ying Hu and Xi Weng and Junlong Jia and Jie Luo and Xien Liu and Ji Wu and Lei Huang},
      year={2024},
      eprint={2402.14289},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```BibTeX
@article{jia2024tinyllava,
  title={TinyLLaVA Factory: A Modularized Codebase for Small-scale Large Multimodal Models},
  author={Jia, Junlong and Hu, Ying and Weng, Xi and Shi, Yiming and Li, Miao and Zhang, Xingjian and Zhou, Baichuan and Liu, Ziyu and Luo, Jie and Huang, Lei and Wu, Ji},
  journal={arXiv preprint arXiv:2405.11788},
  year={2024}
}
```