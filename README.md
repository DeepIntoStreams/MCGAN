# Monte Carlo GAN

Monte Carlo GAN (MCGAN) is a novel framework that incorporates mean squared error (MSE) and the Monte Carlo method into the generative loss function. This innovative generative loss function provides strong supervision to guide the generator training and enhance its performance. 

Authors: Baoren Xiao, Hao Ni

Paper Link: 

This repo contains our implementation of Monte Carlo CGAN on CIFAR-10 and CIFAR-100 datasets using BigGAN and StyleGAN2 as the backbone, the detailed instruction can be founded in each folder where a toy example called Dirac-GAN is also provided. This code base only supports PyTorch.

## Loss function 
The MCGAN use the following loss functions:

$$
\begin{aligned}
\mathcal{L}\_{D}(\psi;\theta)&=\mathbb{E}\_{\mu}\left[f\_1(D^\psi(X))\right]+\mathbb{E}\_{\nu\_\theta}\left[f\_2(D^\psi(X))\right]\\
\mathcal{L}\_{G}(\theta;\phi)&=\mathbb{E}\_{\mu}[|D^\phi(X,Y)-\mathbb{E}\_{\nu\_{\theta}(Y)}[D^\phi(X,Y)]|^2],
\end{aligned}
$$

where $f_1,f_2$ are two functions determined by the specified discriminative loss function. In our paper, we choose Hinge loss and BCE loss as the discriminative loss function and use our proposed method to enhance the generator training. 

## Test metrics on CIFAR-10 using BigGAN

| Method   |   IS ↑ | FID ↓ | IFID ↓ |
| --- | --- | --- | --- | 
| Hinge | 9.61 | 4.43 | 14.60 |
| Hinge + MC | *9.96* | *3.61* | *13.60* |
| BCE | 9.51  | 4.71 | 14.83|
| BCE + MC | 9.94  | 3.93 | 13.72 |

## Test metrics on CIFAR-10 using StyleGAN2

| Method   |   IS ↑ | FID ↓ | IFID ↓ |
| --- | --- | --- | --- | 
| Hinge | 10.186 | 2.248 | 11.404 |
| Hinge + MC | **10.261** | **2.165** | **11.035** |
| BCE | 10.128 | 2.439 | 11.619 |
| BCE + MC | 10.100 | 2.360 | 11.300 |

## Citation
If you find the code helpful, please cite this paper:
```
@inproceedings{....,
  title={....},
  author={......},
  booktitle={....},
  year={...}
}
```
