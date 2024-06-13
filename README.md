# 🐦 Magpie

This is the official repository for "[Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)".

- 🤗 [**Huggingface (Models and Datasets)**](https://huggingface.co/Magpie-Align)
- 🕸️ [**Website**](https://magpie-align.github.io/)
- 📄 [**Technical Report**](https://arxiv.org/abs/2406.08464)


[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2406.08464) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

High-quality instruction data is critical for aligning large language models (LLMs). Although some models, such as Llama-3-Instruct, have open weights, their alignment data remain private, which hinders the democratization of AI. High human labor costs and a limited, predefined scope for prompting prevent existing open-source data creation methods from scaling effectively, potentially limiting the diversity and quality of public alignment datasets. Is it possible to synthesize high-quality instruction data at scale by extracting it directly from an aligned LLM? We present a self-synthesis method for generating large-scale alignment data named Magpie. Our key observation is that aligned LLMs like Llama-3-Instruct can generate a user query when we input only the left-side templates up to the position reserved for user messages, thanks to their auto-regressive nature. We use this method to prompt Llama-3-Instruct and generate 4 million instructions along with their corresponding responses. We perform a comprehensive analysis of the extracted data and select 300K high-quality instances. To compare Magpie data with other public instruction datasets, we fine-tune Llama-3-8B-Base with each dataset and evaluate the performance of the fine-tuned models. Our results indicate that in some tasks, models fine-tuned with Magpie perform comparably to the official Llama-3-8B-Instruct, despite the latter being enhanced with 10 million data points through supervised fine-tuning (SFT) and subsequent feedback learning. We also show that using Magpie solely for SFT can surpass the performance of previous public datasets utilized for both SFT and preference optimization, such as direct preference optimization with UltraFeedback. This advantage is evident on alignment benchmarks such as AlpacaEval, ArenaHard, and WildBench.

## Overview

![Overview](figs/overview.png)

## Toy Example
**Build environment**
```
git clone https://github.com/magpie-align/magpie.git
cd magpie
conda create -n magpie python=3.10
conda activate magpie
pip install -r requirements.txt
```

**Get access to Llama-3 models from Huggingface** 🫨

To login in terminal, enter:
```
huggingface-cli login
```
then enter your Huggingface private key beginning with "hf_".

**Play with Jupyter Notebook**

The toy example can be found in ```demo.ipynb```. Have fun!

## Batched Data Generation

We will release batch & tagging & labeling & sanitization codes soon after June 15th 🥺!