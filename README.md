<h1 align="center">EfficientLLM: Pruning-Aware Pretraining</h1>


<p align="center">
    <a href="https://arxiv.org/abs/2502.06663">
            <img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-2502.06663-b31b1b.svg">
    </a>
    <a href="https://huggingface.co/collections/xrxing/efficientllm-pruning-aware-pretraining-67a8ecc6a49580b647a6184f">
        <img alt="Build" src="https://img.shields.io/badge/HF%20Model-🤗-yellow">
    </a>
</p>

This repository contains the training code and models of EfficientLLM introduced in our work: ["EfficientLLM: Scalable Pruning-Aware Pretraining for Architecture-Agnostic Edge Language Models"](https://arxiv.org/abs/2502.06663).

## News
- Feb 10, 2025: 🚀 EfficientLLM models are publicly available on [HuggingFace](https://huggingface.co/collections/xrxing/efficientllm-pruning-aware-pretraining-67a8ecc6a49580b647a6184f).


## 1. Overview

Modern large language models (LLMs) driven by scaling laws, achieve intelligence emergency in large model sizes. Recently, the increasing concerns about cloud costs, latency and privacy make it an urgent requirement to develop compact edge language models. Distinguished from direct pretraining that bounded by the scaling law, this work proposes the pruning-aware pretraining, focusing on retaining performance of much larger optimized models. It features following characteristics: 1) Data-scalable: we introduce minimal parameter groups in LLM and continuously optimize structural pruning, extending post-training pruning methods like LLM-Pruner and SparseGPT into the pretraining phase. 2) Architecture-agnostic: the LLM architecture is auto-designed using saliency-driven pruning, which is the first time to exceed SoTA human-designed LLMs in modern pretraining. We reveal that it achieves top-quality edge language models, termed EfficientLLM, by scaling up LLM compression and extending its boundary.

<div align=center>
<img width=90% src="https://github.com/Xingrun-Xing2/EfficientLLM/blob/main/imgs/fig2.png"/>
</div>

**Figure 1**: An overview of pruning-aware pretraining. (a) Training loop includes the joint saliency detection and weight optimizing, pruning type selection from pruning space, and second-order weight updating. (b) Traditional post-training pruning can be embedded in the training loop to scale up. (c) Continuous model size compression in pretraining.

## 2. Results on Zero-shot Benchmarks

<div align=center>
<img width=98% src="https://github.com/Xingrun-Xing2/EfficientLLM/blob/main/imgs/table2.PNG"/>
</div>

## 3. Load Huggingface Models

To load a pre-trained model and tokenizer, you can use the following code snippet:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("xrxing/EfficientLLM-469M", use_fast=False)

# Load the model
model = AutoModelForCausalLM.from_pretrained("xrxing/EfficientLLM-469M", trust_remote_code=True, attn_implementation="flash_attention_2")
```

## 4. ToDo List

- [x] Release technical report
- [x] Release Huggingface models
- [ ] Evaluation code
- [ ] Pretraining code
- [ ] Demos and applications


## Contact

Xingrun Xing, CASIA, BAAI (xingxingrun2023@ia.ac.cn)

## Citation
If you find this work useful for your research, please consider citing:
```
@misc{xing2025efficientllm,
      title={EfficientLLM: Scalable Pruning-Aware Pretraining for Architecture-Agnostic Edge Language Models}, 
      author={Xingrun Xing and Zheng Liu and Shitao Xiao and Boyan Gao and Yiming Liang and Wanpeng Zhang and Haokun Lin and Guoqi Li and Jiajun Zhang},
      year={2025},
      eprint={2502.06663},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.06663}, 
}
```

