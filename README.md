# ParaTAA-Diffusion
This is the official repo for the paper "Accelerating Parallel Sampling of Diffusion Models" Tang et al. ICML 2024 \[[paper](https://openreview.net/forum?id=CjVWen8aJL)\].

# Environment

# Use cases
Remark: In the following implementation, we use accelerate package from HuggingFace to implement DDP (Distributed Data Parallelism) for spliting the batch inference across 8 GPUs evenly.

## 1. ParaTAA with DiT
The used DiT models can be found here \[[DiT](https://github.com/facebookresearch/DiT)\].
```
# Running ParaTAA with DiT on 8 GPUs, key parameters below
accelerate launch --num_processes 8 parallel_dit.py \
--timesteps <> \ # number of timesteps for generation
--cfg_scale <> \ # classifier-free guidance
--eta <> \ # eta for DDIM, eta=0 is ODE sampler, eta>0 is SDE sampler
--seed <> \ # random seed
--num_samples <> \ # number of samples to generate 
--window_size <> \ # equivalent to effect batch size
--order <> \ # order of used nonlinear equations
--memory_size <> \ # Parameter for Triangular Anderson Acceleration, determines the number of previous iteration to be used, recommended to be set between 2-5. If set to 1, no Triangular Anderson Acceleration will be used and it reduces to the naive Fixed-Point iteration.
--max_steps <> \ # maximum number of steps for the fixed-point iteration
--model_path <> \ # path to the pretrained DiT model
--vae_path <> \ # path to the pretrained VAE model
--output_path <> \ # path to save the generated samples
--fp16 # whether to use fp16, store_true action

# Example
accelerate launch --num_processes 8 parallel_dit.py --max_steps 10 --fp16
```
After running the above command, you will get the generated samples in the output_path. This command will store all the intermediate samples during the generation, which can help you see how the sample evolves after each step of the fixed-point iteration.

Sample output are provided in the output folder.


## 2. ParaTAA with SD v1.5
The used SD models can be found here \[[SD 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)\]. Most of the parameters are the same as the ones used in the paper, in the following command, we only list the key parameters.
```
# Running ParaTAA with DiT on 8 GPUs, key parameters below
accelerate launch --num_processes 8 parallel_sd.py \
--prompt <> \ # The prompt use for text-to-image generation
--model_path <> \ # The path to the diffuser pipeline of the SD 1.5 model

# Example
accelerate launch --num_processes 8 parallel_sd.py --max_steps 15 --fp16
```
The output will be the same as the one in the case 1.


## 3. ParaTAA with SD v1.5, initializing from existing samples


# If you found this code useful, please cite our paper:

```
@inproceedings{
tang2024accelerating,
title={Accelerating Parallel Sampling of Diffusion Models},
author={Zhiwei Tang and Jiasheng Tang and Hao Luo and Fan Wang and Tsung-Hui Chang},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=CjVWen8aJL}
}
```