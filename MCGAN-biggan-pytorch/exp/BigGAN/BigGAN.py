import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch import autograd
import layers

from DiffAugment_pytorch  import DiffAugment

from sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d

# from template_lib.v2.config import global_cfg
from template_lib.v2.config_cfgnode import global_cfg

# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}
  arch[512] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
               'upsample' : [True] * 7,
               'resolution' : [8, 16, 32, 64, 128, 256, 512],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,10)}}
  arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
  arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
  arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

  return arch

class Generator(nn.Module):
  def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
               G_kernel_size=3, G_attn='64', n_classes=1000,
               num_G_SVs=1, num_G_SV_itrs=1,
               G_shared=True, shared_dim=0, hier=False,
               cross_replica=False, mybn=False,
               G_activation=nn.ReLU(inplace=False),
               G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
               BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
               G_init='ortho', skip_init=False, no_optim=False,
               G_param='SN', norm_style='bn',
               **kwargs):
    super(Generator, self).__init__()
    # Channel width mulitplier
    self.ch = G_ch
    # Dimensionality of the latent space
    self.dim_z = dim_z
    # The initial spatial dimensions
    self.bottom_width = bottom_width
    # Resolution of the output
    self.resolution = resolution
    # Kernel size?
    self.kernel_size = G_kernel_size
    # Attention?
    self.attention = G_attn
    # number of classes, for use in categorical conditional generation
    self.n_classes = n_classes
    
    # Use shared embeddings?                                                         ##################################What is shared embedding 
    self.G_shared = G_shared
    
    # Dimensionality of the shared embedding? Unused if not using G_shared
    self.shared_dim = shared_dim if shared_dim > 0 else dim_z
    # Hierarchical latent space?
    self.hier = hier                                       ################################################# Flase 
    # Cross replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # nonlinearity for residual blocks
    self.activation = G_activation
    # Initialization style
    self.init = G_init
    # Parameterization style
    self.G_param = G_param
    # Normalization style
    self.norm_style = norm_style
    # Epsilon for BatchNorm?
    self.BN_eps = BN_eps
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # fp16?
    self.fp16 = G_fp16
    # Architecture dict
    self.arch = G_arch(self.ch, self.attention)[resolution]

    # If using hierarchical latents, adjust z
    if self.hier:
      # Number of places z slots into
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      # Recalculate latent dimensionality for even splitting into chunks
      self.dim_z = self.z_chunk_size *  self.num_slots
    else:
      self.num_slots = 1
      self.z_chunk_size = 0

    # Which convs, batchnorms, and linear layers to use
    if self.G_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
    else:
      self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear
    
    # We use a non-spectral-normed embedding here regardless;
    # For some reason applying SN to G's embedding seems to randomly cripple G
    self.which_embedding = nn.Embedding ###non-spectral-normed embedding
    
    bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared              ###################################bn_linear is used for 
                 else self.which_embedding)                                                     #######batch norm linear layer is actually embedding
                 
    # Add Class conditional batch norm 
    self.which_bn = functools.partial(layers.ccbn,
                          which_linear=bn_linear,
                          cross_replica=self.cross_replica,
                          mybn=self.mybn,
                          input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                      else self.n_classes),
                          norm_style=self.norm_style,
                          eps=self.BN_eps)


    # Prepare model
    # If not using shared embeddings, self.shared is just a passthrough
    self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared 
                    else layers.identity())
    # First linear layer
    self.linear = self.which_linear(self.dim_z // self.num_slots,                                 ##############################  self.num_slots =1 by default 
                                    self.arch['in_channels'][0] * (self.bottom_width **2))

    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    # while the inner loop is over a given block
    self.blocks = []                                                                                   ##########################Buildinng the ResBloack for G
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['out_channels'][index],
                             which_conv=self.which_conv,
                             which_bn=self.which_bn,
                             activation=self.activation,
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if self.arch['upsample'][index] else None))]]

      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:                     #######################################For CIFAR10 case  Attention is False###########################################
        print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # output layer: batchnorm-relu-conv.
    # Consider using a non-spectral conv here
    self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                cross_replica=self.cross_replica,
                                                mybn=self.mybn),
                                    self.activation,
                                    self.which_conv(self.arch['out_channels'][-1], 3))

    # Initialize weights. Optionally skip init for testing. True if using ema
    if not skip_init:                                                     ######################################## skip_init set as False by default ##################################################
      self.init_weights()

    # Set up optimizer
    # If this is an EMA copy, no need for an optim, so just return now
    if no_optim:  
      return
      
    self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
    if G_mixed_precision: ######################################################################################## skip_init set as False by default ##################################################
      print('Using fp16 adam in G...')
      import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0,
                           eps=self.adam_eps)
    else:
      weight_decay = global_cfg.Generator.get('weight_decay')                              ######################## What is the global  
      optim_type = global_cfg.get('optim_type', 'adam')                                    #############
      
      
      if optim_type.lower() == 'adam':                                                #######################Adam by default 
        self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                betas=(self.B1, self.B2), weight_decay=weight_decay,
                                eps=self.adam_eps)
      elif optim_type.lower() == 'adams':
        from .adams import AdamS
        self.optim = AdamS(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=weight_decay,
                           eps=self.adam_eps, amsgrad=False)
      else:
        assert 0
    #self.scheduler=optim.lr_scheduler.ExponentialLR(self.optim,gamma=0.9999)
    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:                                     
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for G''s initialized parameters: %d' % self.param_count)

  # Note on this forward function: we pass in a y vector which has
  # already been passed through G.shared to enable easy class-wise
  # interpolation later. If we passed in the one-hot and then ran it through
  # G.shared in this forward function, it would be harder to handle.
  def forward(self, z, y):
    # If hierarchical, concatenate zs and ys
    if self.hier:                              ##################################################Flase 
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.blocks) #################################Here y is probaly a label rather than one-hot
      
    # First linear layer
    h = self.linear(z)
    # Reshape
    h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
    
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
        h = block(h, ys[index])
        
    # Apply batchnorm-relu-conv-tanh at output
    return torch.tanh(self.output_layer(h))                 #############################################with tanh activation function the output image has value in [-1,1]
    
    
  
  

# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64',ksize='333333', dilation='111111'):
  arch = {}
  arch[256] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[128] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[64]  = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,7)}}
  arch[32]  = {'in_channels' :  [3] + [item * ch for item in [4, 4, 4]],
               'out_channels' : [item * ch for item in [4, 4, 4, 4]],
               'downsample' : [True, True, False, False],
               'resolution' : [16, 16, 16, 16],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,6)}}
  return arch

class Discriminator(nn.Module):

  def __init__(self, D_ch=64, D_wide=True, resolution=128,
               D_kernel_size=3, D_attn='64', n_classes=1000,
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
               D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
               SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
               D_init='ortho', skip_init=False, D_param='SN', **kwargs):
    super(Discriminator, self).__init__()
    # Width multiplier
    self.ch = D_ch
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide = D_wide
    # Resolution
    self.resolution = resolution
    # Kernel size
    self.kernel_size = D_kernel_size
    # Attention?
    self.attention = D_attn
    # Number of classes
    self.n_classes = n_classes
    # Activation
    self.activation = D_activation
    # Initialization style
    self.init = D_init
    # Parameterization style
    self.D_param = D_param
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # Fp16?
    self.fp16 = D_fp16
    # Architecture
    self.arch = D_arch(self.ch, self.attention)[resolution]

    self.weight_decay = global_cfg.Discriminator.weight_decay

    # Which convs, batchnorms, and linear layers to use
    # No option to turn off SN in D right now
    if self.D_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_embedding = functools.partial(layers.SNEmbedding,
                              num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                              eps=self.SN_eps)
    # Prepare model
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    #self.arch['in_channels'][0]+=self.n_classes
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       wide=self.D_wide,
                       activation=self.activation,
                       preactivation=(index > 0),
                       downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:          ###########################################No attention for CIFAR10
        print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                             self.which_conv)]
    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output 
    output_dim = self.n_classes + 2 if global_cfg.loss.losstype=='omni' else 1 
    self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)

    # Embedding for projection discrimination
    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

    # Initialize weights
    if not skip_init:                                                            ############################### skip_init set as False by default 
      self.init_weights()

    # Set up optimizer
    self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
    
    if D_mixed_precision:                                               #########################################################set False  by default
      print('Using fp16 adam in D...')
      import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=self.weight_decay, eps=self.adam_eps)
    else:
      optim_type = global_cfg.get('optim_type', 'adam')
      if optim_type.lower() == 'adam':                            #########################################################set Adam by default
        self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                               betas=(self.B1, self.B2), weight_decay=self.weight_decay, eps=self.adam_eps)
      elif optim_type.lower() == 'adams':
        from .adams import AdamS
        self.optim = AdamS(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=self.weight_decay,
                           eps=self.adam_eps, amsgrad=False)
      else:
        assert 0
    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0
    pass

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)
    
  def Cat2Code(self,label:torch.Tensor,n_class:int):
    batch=len(label)
    label_indx= torch.from_numpy(np.r_[np.arange(batch)[None,:],label.cpu().numpy()[None,:]]).long().to(label.device)
    code=torch.zeros(batch,n_class).to(label.device)
    code[tuple(label_indx)]=1
    return code
    
  def forward(self, x, y):
    # Stick x into h for cleaner for loops without flow control
    #y=self.Cat2Code(y,n_class=self.n_classes)
    #x =torch.cat([x, y[:,:,None,None]*torch.ones([x.shape[0],y.shape[1], x.shape[2], x.shape[3],]).to(x.device)], dim=1)
    h = x
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:    ############################################# #############probally change the way to feed in the label 
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3])  #relu here
    #Project the features into hypersphere
    #h = h/(torch.sqrt(h.pow(2).sum(1,keepdim=True))+1e-4)
    
    # Get initial class-unconditional output
    out = self.linear(h) ############### The current output dim is 1
    # Get projection of final featureset onto class vectors and add to evidence
    out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
    #out=(torch.exp(out)+1).pow(-1)
    return out

# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
  def __init__(self, G, D):
    super(G_D, self).__init__()
    self.G = G
    self.D = D
    self.mc_size=10 
    self.mc_init=10
    self.mc_ceil=100
    self.mc_scalar=(self.mc_ceil-self.mc_init)/2
    self.count=0
  
        
  def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
              split_D=False,policy=None,mc=False,mc_size=10,CR=False, CR_augment=None,SQ=False):              
############Generate fake img  G_z given gy #####################################
    batch_size=z.shape[0]
    with torch.set_grad_enabled(train_G):
      if not mc:
      # Get Generator output given noise
          G_z= self.G(z, self.G.shared(gy)) if train_G else self.G(z, self.G.shared(dy))
      else:
          x=DiffAugment(x, policy=policy) 
          D_real=self.D(x,dy) 
          z_mc=torch.randn(self.mc_init*batch_size,self.G.dim_z,device=dy.device)
          dy_mc=dy.repeat(self.mc_init)
          G_z_mc=self.G(z_mc, self.G.shared(dy_mc))
          G_z_mc=DiffAugment(G_z_mc,policy=policy)
          D_mc=self.D(G_z_mc,dy_mc).reshape(self.mc_init,batch_size,-1).clamp(min=-1.5,max=1.5)
          D_fake_mc=D_mc.mean(0)
          assert D_fake_mc.shape==D_real.shape
          if SQ:
              D_fake_mc_sq=torch.clamp(D_mc,min=-1.5,max=1.5).pow(2).mean(0)
          else:
              D_fake_mc_sq=None
          return D_fake_mc,D_real,D_fake_mc_sq
      # Cast as necessary
      if self.G.fp16 and not self.D.fp16:
        G_z = G_z.float()
      if self.D.fp16 and not self.G.fp16:
        G_z = G_z.half()
        
    D_input = torch.cat([G_z, x], 0) if x is not None else G_z
    D_class = torch.cat([dy, dy], 0) if dy is not None else gy
    D_input = DiffAugment(D_input, policy=policy)
    #CR
    if CR:
      if CR_augment:
        x_CR_aug = torch.split(D_input, [G_z.shape[0], x.shape[0]])[1]
        if CR_augment.startswith('flip,'):
                    x_CR_aug = torch.where(torch.randint(0, 2, size=[x_CR_aug.size(
                        0), 1, 1, 1], device=x_CR_aug.device) > 0, x_CR_aug.flip(3), x_CR_aug)
        x_CR_aug = DiffAugment(x_CR_aug, policy=CR_augment.replace('flip,', ''))
        D_input = torch.cat([D_input, x_CR_aug], 0)
      else:
        D_input = torch.cat([D_input, x], 0)
      D_class = torch.cat([D_class, dy], 0)
    # Get Discriminator output
    D_out = self.D(D_input, D_class)
    # if CR return X_CR_aug
    if x is not None:
        if CR:
                return torch.split(D_out, [G_z.shape[0], x.shape[0], x.shape[0]])
        else:
                return torch.split(D_out, [G_z.shape[0], x.shape[0]])
    else:
        if return_G_z:
          return D_out, G_z
        else:
          return D_out
