import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import random


def multilabel_categorical_crossentropy(y_true, y_pred, margin=0., gamma=1.):
  """
  y_true: positive=1, negative=0, ignore=-1

  """
  y_true = y_true.clamp(-1, 1)
  if len(y_pred.shape) > 2:
    y_true = y_true.view(y_true.shape[0], 1, 1, -1)
    _, _, h, w = y_pred.shape
    y_true = y_true.expand(-1, h, w, -1)
    y_pred = y_pred.permute(0, 2, 3, 1)

  y_pred = y_pred + margin
  y_pred = y_pred * gamma

  y_pred[y_true == 1] = -1 * y_pred[y_true == 1]
  y_pred[y_true == -1] = -1e12

  y_pred_neg = y_pred.clone()
  y_pred_neg[y_true == 1] = -1e12

  y_pred_pos = y_pred.clone()
  y_pred_pos[y_true == 0] = -1e12

  zeros = torch.zeros_like(y_pred[..., :1])
  y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
  y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
  neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
  pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
  return neg_loss + pos_loss


class OmniLoss(object):

  def __init__(self, default_label=0, margin=0., gamma=1.):
    self.default_label = default_label
    self.margin = margin
    self.gamma = gamma
    pass

  @staticmethod
  def get_one_hot(label_list, one_hot, b, filled_value=0):

    for label in label_list:
      if isinstance(label, int):
        label = torch.empty(b, dtype=torch.int64, device=one_hot.device).fill_(label)
      one_hot.scatter_(dim=1, index=label.view(-1, 1), value=filled_value)
    return one_hot

  def __call__(self, pred, positive=None, negative=None, default_label=None, margin=None, gamma=None):
    default_label = self.default_label if default_label is None else default_label
    margin = self.margin if margin is None else margin
    gamma = self.gamma if gamma is None else gamma

    b, nc = pred.shape[:2]
    label_onehot = torch.empty(b, nc, dtype=torch.int64, device=pred.device).fill_(default_label)

    if positive is not None:
      label_onehot = OmniLoss.get_one_hot(label_list=positive, one_hot=label_onehot, b=b, filled_value=1)

    if negative is not None:
      label_onehot = OmniLoss.get_one_hot(label_list=negative, one_hot=label_onehot, b=b, filled_value=0)

    loss = multilabel_categorical_crossentropy(
      y_true=label_onehot, y_pred=pred, margin=margin, gamma=gamma)
    loss_mean = loss.mean()
    return loss_mean

  @staticmethod
  def test_case():
    b, nc = 32, 101
    pred = torch.rand(b, nc).cuda().requires_grad_()
    y = torch.randint(0, nc-1, (b, )).cuda()

    omni_loss = OmniLoss()
    D_loss_real = omni_loss(pred=pred, positive=(y, nc-1))

    D_loss_fake = omni_loss(pred=pred, positive=None, negative=(y, nc-1))

    adv_loss = omni_loss(pred=pred, positive=(y, nc-1))

    D_loss_real = omni_loss(pred=pred, positive=(y, nc - 1), default_label=0)

    # b, nc, h, w = 32, 101, 8, 8
    # pred = torch.rand(b, nc, h, w).cuda().requires_grad_()
    # y = torch.randint(0, nc - 1, (b, h, w)).cuda()
    # D_loss_real = omni_loss(pred=pred, positive=(y, nc - 1))
    pass


class W1Distance(object):
  def __init__(self,):
    pass
  def __call__(self, D_fake,D_real=None,GP=None,reg=10,train_G=False):
    #compute loss for Discriminator
    if D_real is not None and GP is not None:
        loss =-D_real.mean() + D_fake.mean()
        #compute penalty
        loss += reg*GP
    #compute loss for Generator 
    else:
        loss=-D_fake.mean()
    return loss


class MSELoss(object):
  def __init__(self,):
    self.BCE = nn.BCELoss()

    pass
  def __call__(self, D_fake,D_real,a=1,b=0,train_G=False,D_fake_sq=None,SQ=0):
    #compute loss for Discriminator
    if not train_G:
        loss = nn.ReLU()(1.0 - D_real).mean() + nn.ReLU()(1.0 + D_fake).mean()
        #compute penalty
    #compute loss for Generator 
    else:
    #For Generaortrain_G=False,
        if SQ>0:
          loss=-D_fake.mean()+(D_real-D_fake).pow(2).mean()+SQ*(torch.clamp(D_real,min=-1,max=1).pow(2)-D_fake_sq).pow(2).mean()#D_fake_mc
        else:
          loss=-D_fake.mean()+(D_real-D_fake).pow(2).mean()
    return loss

class HingeLoss(object):
  def __init__(self,):
    pass
  def __call__(self, D_fake,D_real=None,a=1,b=0,train_G=False):
    #compute loss for Discriminator
    if not train_G:
        loss = nn.ReLU()(1.0 - D_real).mean() + nn.ReLU()(1.0 + D_fake).mean()
        #compute penalty
    #compute loss for Generator 
    else:
    #For Generaor
        loss=-D_fake.mean()
    return loss
    
class NonsaturatingLoss(object):
    def __init__(self,):
      pass
    def __call__(self, D_fake,D_real=None,train_G=False):
    #compute loss for Discriminator
      if not train_G:
        loss =F.softplus(- D_real).mean() + F.softplus(D_fake).mean()
        #compute penalty
    #compute loss for Generator 
      else:
    #For Generaor
        loss=F.softplus(- D_fake).mean() 
      return loss
    
Adversarial_loss={'omni':OmniLoss,'hinge':HingeLoss,'w1':W1Distance,'ns':NonsaturatingLoss}

# Generative conditioning loss
class MSELoss_G(object):
  def __init__(self,):
    pass
  def __call__(self, D_fake,D_real,D_fake_sq=None,SQ=0):
    #compute loss for Discriminator
    if SQ>0:
          loss=(D_real-D_fake).pow(2).mean()+SQ*(torch.clamp(D_real,min=-1,max=1).pow(2)-D_fake_sq).pow(2).mean()#D_fake_mc
    else:
          loss=(D_real-D_fake).pow(2).mean()
    return loss
    
    
if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '7'
  OmniLoss.test_case()

