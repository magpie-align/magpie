<!-- # 🐦 Magpie -->

![Magpie](figs/magpie_logo.png)

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2406.08464) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://huggingface.co/Magpie-Align) [![Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/davanstrien/magpie)

This is the official repository for "[Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)".

- 🤗 [**Huggingface (Models and Datasets)**](https://huggingface.co/Magpie-Align)
- 🕸️ [**Website**](https://magpie-align.github.io/)
- 📄 [**Technical Report**](https://arxiv.org/abs/2406.08464)

You can try the no-code Magpie demo [🤗 here](https://huggingface.co/spaces/davanstrien/magpie) to generate instruction-response pairs. Thanks a lot for the quick implementation from @davanstrien!

## Magpie Supports

Currently, Magpie has been tested on the **Llama-3**, **Qwen2**, and **Phi 3** series. Feel free to submit a pull request to [`configs/model_configs.json`](configs/model_configs.json) with more model support.

|Model Family | Magpie | Magpie Script | Dataset |
|-------------|:------:|:-------|:-------|
| [Llama 3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6)     | ✅ | [8B](scripts/magpie-llama3-8b.sh),[70B](scripts/magpie-llama3-70b.sh) | [8B](https://huggingface.co/collections/Magpie-Align/magpie-air-6666b11a32021655a27f86c0),[70B](https://huggingface.co/collections/Magpie-Align/magpie-pro-6666b0e713e5f5c09554876f)
| [Qwen2](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f)     | ✅ | [7B](scripts/magpie-qwen2-7b.sh),[72B](scripts/magpie-qwen2-72b.sh) | [72B](https://huggingface.co/collections/Magpie-Align/magpie-qwen2-6679fb5deb164e55849b4907)
| [Phi 3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3)     | ✅ | [mini](scripts/magpie-phi3mini.sh),[small](scripts/magpie-phi3small.sh),[medium](scripts/magpie-phi3medium.sh) | [medium](https://huggingface.co/collections/Magpie-Align/magpie-phi3-667a7a45f1a406cd61685d64)
| [Gemma-2](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)    | ⭕️ | [9B](magpie-gemma2-9b.sh) [27B](/home/nsl/zhangchen/magpie/scripts/magpie-gemma2-27b.sh)
| [Gemma-1.1](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b)    | ⭕️ | [7B](scripts/magpie-gemma7b.sh)
| [Llama 2](https://huggingface.co/collections/meta-llama/llama-2-family-661da1f90a9d678b6f55773b)   | ⭕️ | [7B](scripts/magpie-llama2-7b.sh),[70B](scripts/magpie-llama2-70b.sh)
| [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)   | ⭕️ | [7B](scripts/magpie-mistral7b.sh)
| [Yi](https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8)    | ⭕️ | [34B](scripts/magpie-yi34b.sh)

- ✅: Works so great!
- ⭕️: Partially work. We can get something interesting, but may apply a powerful filter and/or a logits processor.
- ❌: Not work.
- ❓: Untested.

We hope Magpie can contribute to the democratization of AI. With your help, we can create more data and enhance the transparency of model alignment processes!

### Dataset Generation Queue
- Qwen/Qwen2-7B-Instruct (3M) - Generating
- google/gemma-2-27b-it (1M) - Waiting for vllm support


## Abstract

High-quality instruction data is critical for aligning large language models (LLMs). Although some models, such as Llama-3-Instruct, have open weights, their alignment data remain private, which hinders the democratization of AI. High human labor costs and a limited, predefined scope for prompting prevent existing open-source data creation methods from scaling effectively, potentially limiting the diversity and quality of public alignment datasets. Is it possible to synthesize high-quality instruction data at scale by extracting it directly from an aligned LLM? We present a self-synthesis method for generating large-scale alignment data named Magpie. Our key observation is that aligned LLMs like Llama-3-Instruct can generate a user query when we input only the left-side templates up to the position reserved for user messages, thanks to their auto-regressive nature. We use this method to prompt Llama-3-Instruct and generate 4 million instructions along with their corresponding responses. We perform a comprehensive analysis of the extracted data and select 300K high-quality instances. To compare Magpie data with other public instruction datasets, we fine-tune Llama-3-8B-Base with each dataset and evaluate the performance of the fine-tuned models. Our results indicate that in some tasks, models fine-tuned with Magpie perform comparably to the official Llama-3-8B-Instruct, despite the latter being enhanced with 10 million data points through supervised fine-tuning (SFT) and subsequent feedback learning. We also show that using Magpie solely for SFT can surpass the performance of previous public datasets utilized for both SFT and preference optimization, such as direct preference optimization with UltraFeedback. This advantage is evident on alignment benchmarks such as AlpacaEval, ArenaHard, and WildBench.

## Overview

![Overview](figs/overview.png)

## Installation

**Build environment**
```
git clone https://github.com/magpie-align/magpie.git
cd magpie
conda create -n magpie python=3.10
conda activate magpie
pip install -r requirements.txt
```

**Get access to Llama-3 models from 🤗 Huggingface**

You can apply for Llama-3 model access [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). To login in the terminal, enter:
```
huggingface-cli login
```
then enter your Huggingface private key beginning with "hf_".

## Toy Example

**Play with Jupyter Notebook**

The toy example can be found in [`demo.ipynb`](demo.ipynb). Have fun! 

<a target="_blank" href="https://colab.research.google.com/github/magpie-align/magpie/blob/main/demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Batched Data Generation
We use Llama-3-8B-Instruct as an example to demonstrate the batched data generation process. To run batched generation, you can simply run:
```
cd scripts
bash magpie.sh
```
The script will generate both instructions and responses in the data folder. It has been tested on an RTX 4090 24G GPU. If you are using GPUs with less memory, consider implementing [quantization](https://docs.vllm.ai/en/latest/quantization/fp8.html).

We also provide scripts for other models in the [`scripts`](scripts) folder. You can use [this](#magpie-supports) navigation to find specific Magpie scripts. Note that for model sizes greater than 8B, you may need 4*A100 GPUs to run the scripts.

### Batched Multi-turn Data Generation \[Optional\]
After generating instruction-response pairs, you can extend them to multi-turn conversations. To do so, simply run the following command:
```
bash magpie-multi-turn.sh ***_ins_res.json
```
where `***_ins_res.json` is the single-turn instruction-response pairs generated in the previous step.

## Dataset Filtering
### 1. Tagging
To tag the generated instruction-response pairs, you can run:
```
cd scripts
bash unitag.sh ***_ins_res.json all
```
This script will automatically generate quality, difficulty, task category, safety, reward, and language for the generated dataset. You can also generate one tag at a time. For example, if you just want to generate the safety label using device 0, you can run:
```
cd scripts
bash unitag.sh ***_ins_res.json safety 0
```
### 2. Data Concatenation and Converting
You may generate datasets with different generation configurations. We provide a Jupyter notebook [here](data/data_concatenation.ipynb) for concatenating all datasets and converting them to ShareGPT format, which is fully supported by [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) for fine-tuning.

### 3. Removing Repetition
Once you have a full dataset converted to ShareGPT format, you can calculate the minimum neighbor distance of each instruction and remove repetitions. To do so, run:
```
cd exp
python gen_dis.py --input_file ***_sharegpt.jsonl
```
where `***_sharegpt.jsonl` is the dataset path obtained in the previous step. The Python script will take care of building the FAISS index and calculating the minimum distance. 

### 4. Design and Apply Your Filter
We provide a Jupyter notebook [here](data/data_filter.ipynb) for simple filtering. You can adjust the filtering parameters to design and apply your own filter based on your needs.

## Fine-tuning
Please refer to the [fine-tune](fine-tune/) directory for instructions.

## Citation

If you find the model, data, or code useful, please cite our paper:
```
@misc{xu2024magpie,
    title={Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing}, 
    author={Zhangchen Xu and Fengqing Jiang and Luyao Niu and Yuntian Deng and Radha Poovendran and Yejin Choi and Bill Yuchen Lin},
    year={2024},
    eprint={2406.08464},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
