# Fine-tune your Llama-3-8B using Axolotl and Magpie

## Requirements
### Hardware
- 4 GPU @ 80GB VRAM

If you have fewer GPUs, please add `gradient_accumulation_steps` in the config file accordingly.

If you are using GPUs with less VRAM, please consider using `deepspeed zero3` or `FSDP`.

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

## Supervised Fine-tuning
Go to the directory where you placed the YAML file, then run the following command:
```
accelerate launch -m axolotl.cli.train llama3-base-magpie.yaml
```