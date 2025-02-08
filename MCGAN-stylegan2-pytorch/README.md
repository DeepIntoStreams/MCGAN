# MCGAN for StyleGAN2 

This repo is implemented upon [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) with minimal modifications to implement MCGAN algorithm in PyTorch. Please check the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) README for the dependencies and the other usages of this codebase.


## Pre-Trained Models

The following commands are an example of generating images with our pre-trained 100-shot Obama model. See [here](https://data-efficient-gans.mit.edu/models/) for a list of our provided pre-trained models. The code will automatically convert a TensorFlow StyleGAN2 model to the compatible PyTorch version; you may also use `legacy.py` to do this manually.
```bash
python generate.py --outdir=out --seeds=1-16 --network=https://data-efficient-gans.mit.edu/models/DiffAugment-stylegan2-100-shot-obama.pkl

python generate_gif.py --output=obama.gif --seed=0 --num-rows=1 --num-cols=8 --network=https://data-efficient-gans.mit.edu/models/DiffAugment-stylegan2-100-shot-obama.pkl
```

## Train MCGAN
### CIFAR-10
To train an MCGAN on CIFAR-10 dataset
Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/cifar-10-python.tar.gz --dest=~/datasets/cifar10.zip
```

then run the following line:
```bash
python train.py --outdir=training-runs/MCGAN --data=datasets/cifar10.zip --gpus=4 --mc_size=4 
```

### CIFAR-10
To train an MCGAN on CIFAR-10 dataset
Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/cifar-10-python.tar.gz --dest=~/datasets/cifar10.zip
```

then run the following line:
```bash
python train.py --outdir=training-runs/MCGAN --data=datasets/cifar10.zip --gpus=4 --mc_size=4 
``` 
### FFHQ256
Run the following line for 4 gpus:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --outdir=training-runs/ffhq256 --data=data/ffhq256x256.zip --cfg=stylegan2 --gpus=4 --batch=32 --gamma=1 --mirror=1 --glr=0.0025 --dlr=0.0025 --cbase=16384 --mc_size 1
```

### Imagenet64
Run the following line for 4 gpus:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --outdir=training-runs/imagenet64 --data=data/imagenet64.zip --cfg=stylegan2 --gpus=4 --batch=64 --gamma=0.0128 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --cond=1 --snap=100 --mc_size 1
```

### LSUN bedroom 256:
Run the following line for 4 gpus:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --outdir=training-runs/lsunbed --data=data/lsun_bedroom_256 --cfg=stylegan2 --gpus=4 --batch=32 --gamma=10 --mirror=1 --glr=0.0025 --dlr=0.0025 --cbase=16384 --mc_size 1
```








