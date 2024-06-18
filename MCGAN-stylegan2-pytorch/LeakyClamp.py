import torch 
from  torch.nn import  Module,LeakyReLU
from torch import Tensor

class LeakyClamp(Module):
    r"""Applies the element-wise function:
    .. math::
        \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)
    or
    .. math::
        \text{LeakyReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{negative\_slope} \times x, & \text{ otherwise }
        \end{cases}
    Args:
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input
    Examples::
        >>> LeakyClamp(ub=1.1,lb=-1.1,negative_slope=-0.9)
        >>> output = m(input)
    Comments:: the learning rate outside the [lb,ub] is given by (1+negative_slope)/2
    """
    __constants__ = ['inplace', 'negative_slope']
    inplace: bool
    negative_slope: float

    def __init__(self, ub:float,  lb:float , scale: float,negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super(LeakyClamp, self).__init__()
        self.ub=ub
        self.lb=lb
        self.negative_slope = negative_slope
        self.inplace = inplace
        self.scale = scale

    def forward(self, input: Tensor) -> Tensor:
        input = self.scale*input
        if self.inplace:
          output = .5*(-torch._C._nn.leaky_relu_(self.ub-input, self.negative_slope)+self.ub)+.5*(torch._C._nn.leaky_relu_(input-self.lb, self.negative_slope)+self.lb)
        else:
          output = .5*(-torch._C._nn.leaky_relu(self.ub-input, self.negative_slope)+self.ub)+.5*(torch._C._nn.leaky_relu(input-self.lb, self.negative_slope)+self.lb)#not inplace 
        return output

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)
    #clamp=LeakyClamp(ub=1.2,lb=-1.2,negative_slope=0)
