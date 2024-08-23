# Scripts to train Magpie models

## Requirements
### Hardware
- 4 GPU @ 80GB VRAM

If you have fewer GPUs, please add `gradient_accumulation_steps` in the config file accordingly.

If you are using GPUs with less VRAM, please consider using `deepspeed zero3` or `FSDP`.

### Software

We use [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for supervised fine-tuning, and [alignment-handbook](https://github.com/huggingface/alignment-handbook) for DPO. 

## Supervised Fine-tuning

### Install Axolotl
Environment: Python >=3.10 and Pytorch >=2.1.1. You can setup the environment using Conda.

**Note**: We found that the latest Axolotl may encounter some bugs during the training, so we use [this commit](https://github.com/OpenAccess-AI-Collective/axolotl/commit/7c2bf3091f5e73c787afe839dfdcc8220b770a1a) in our experiments. Please also manually install FastChat for the latest Llama3 conversation template support.

```bash
git clone https://github.com/lm-sys/FastChat
cd FastChat
pip install -e .
cd ../

git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
git reset --hard 7c2bf3091f5e73c787afe839dfdcc8220b770a1a

pip3 install packaging ninja
pip3 install -e '.[flash-attn,deepspeed]'
```

**Note 2:** They have fixed that. Please use the latest version.

### Run
Go to the directory where you placed the SFT YAML file, then run the following command:
```
accelerate launch -m axolotl.cli.train your_config_name.yaml
```

## DPO

### Install Alignment Handbook

```bash
conda create -n handbook python=3.10 && conda activate handbook

git clone https://github.com/huggingface/alignment-handbook.git
cd alignment-handbook
python -m pip install .

python -m pip install flash-attn --no-build-isolation
```

### Run

Please change `num_processes` if you are not using 4 GPUs.

```
ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 4 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py your_config_name.yaml
```

## Magpie Recipes

- [Llama-3-8B-Magpie-Align-SFT-v0.1](Llama-3-8B-Magpie-Align-SFT-v0.1)
- [Llama-3-8B-Magpie-Align-SFT-v0.2](Llama-3-8B-Magpie-Align-SFT-v0.2)
- [Llama-3-8B-Magpie-Align-SFT-v0.3](Llama-3-8B-Magpie-Align-SFT-v0.2)
- [Llama-3-8B-Magpie-Align-v0.1](Llama-3-8B-Magpie-Align-v0.1)
- [Llama-3-8B-Magpie-Align-v0.2](Llama-3-8B-Magpie-Align-v0.2)
- [Llama-3-8B-Magpie-Align-v0.3](Llama-3-8B-Magpie-Align-v0.3)
- [Llama-3.1-8B-Magpie-Align-SFT-v0.1](Llama-3.1-8B-Magpie-Align-SFT-v0.1)
- [Llama-3.1-8B-Magpie-Align-SFT-v0.2](Llama-3.1-8B-Magpie-Align-SFT-v0.2)
- [Llama-3.1-8B-Magpie-Align-v0.1](Llama-3.1-8B-Magpie-Align-v0.1)
- [Llama-3.1-8B-Magpie-Align-v0.2](Llama-3.1-8B-Magpie-Align-v0.2)