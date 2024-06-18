# MCGAN for StyleGAN2 

This repo is implemented upon [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) with minimal modifications to implement MCGAN algorithm in PyTorch. Please check the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) README for the dependencies and the other usages of this codebase.


## Pre-Trained Models

The following commands are an example of generating images with our pre-trained 100-shot Obama model. See [here](https://data-efficient-gans.mit.edu/models/) for a list of our provided pre-trained models. The code will automatically convert a TensorFlow StyleGAN2 model to the compatible PyTorch version; you may also use `legacy.py` to do this manually.
```bash
python generate.py --outdir=out --seeds=1-16 --network=https://data-efficient-gans.mit.edu/models/DiffAugment-stylegan2-100-shot-obama.pkl

python generate_gif.py --output=obama.gif --seed=0 --num-rows=1 --num-cols=8 --network=https://data-efficient-gans.mit.edu/models/DiffAugment-stylegan2-100-shot-obama.pkl
```

## Train MCGAN

To train an MCGAN on CIFAR-10 dataset, use the following line:
```bash
python train.py --outdir=training-runs/MCGAN --data=datasets/cifar10.zip --gpus=4 --mc_size=4 
```


