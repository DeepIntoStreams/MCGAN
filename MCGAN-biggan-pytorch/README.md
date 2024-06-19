## MCGAN - BigGAN

This implementation is based on [Omni-GAN-PyTorch](https://github.com/PeterouZh/Omni-GAN-PyTorch) with minimal modifications to incoporate MCGAN algorithm in BigGAN. Please check the [Omni-GAN-PyTorch](https://github.com/PeterouZh/Omni-GAN-PyTorch) README for the dependencies and the other usages of this codebase.
In particular, it contains the code for the CIFAR10 and CIFAR100 experiments. </br >

## Let's go!

### Environments

```bash
git clone --recursive https://github.com/Baoren1996/MCGAN.git
cd MCGAN
conda create -y --name mcgan python=3.6.7 
conda activate mcgan
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
pip install -r requirements.txt
```

### Prepare FID statistics files

The FID statistics files calculated on CIFAR10 and CIFAR100 are provided by  [Omni-GAN-PyTorch](https://github.com/PeterouZh/Omni-GAN-PyTorch) at [OneDrive](https://sjtueducn-my.sharepoint.com/:f:/g/personal/zhoupengcv_sjtu_edu_cn/Ek0QSX1UhylDjVYdmXYxRtcBMLs54AYD4E3CwZlWBXZmPA?e=BzWa9D). Download them and put them into the `datasets` dir. 

### Train the model 
- CIFAR10

For CIFAR10 experiments, you can specify the GPU to be used by setting the CUDA_VISIBLE_DEVICES=0 or 1,2,3, or multiple IDs if GPU pararelle is available.
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./:./BigGAN_PyTorch_1_lib 
export LD_LIBRARY_PATH=$HOME/.keras/envs/cuda-10.0/lib64:$HOME/.keras/envs/cudnn-10.0-linux-x64-v7.6.5.32/lib64 
python exp/BigGAN/train.py \
  --tl_config_file configs/hinge_cifar10.yaml \
  --tl_command train_HingeGAN \
  --tl_outdir results/train_MCGAN_cifar10 \
  --tl_opts GAN_metric.tf_fid_stat datasets/fid_stats_tf_cifar10_train_32.npz \
            GAN_metric.tf_inception_model_dir datasets/tf_inception_model \
            args.data_root datasets/cifar10
```

## Acknowledgments

- BigGAN implemented from [https://github.com/ajbrock/BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch).
- Multi-label classification loss derived by [Jianlin Su](https://kexue.fm/archives/7359).
- Detectron2 library [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2).











