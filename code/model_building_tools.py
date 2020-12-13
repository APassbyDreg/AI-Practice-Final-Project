import torch
from torch import nn


def __norm2d(norm_type, n_out):
    if norm_type == "batch":
        return [nn.BatchNorm2d(n_out)]
    elif norm_type == "instance":
        return [nn.InstanceNorm2d(n_out)]
    elif norm_type == "none":
        return []
    else:
        raise NotImplementedError(f"input norm_type {norm_type} is not implemented")


def __norm1d(norm_type, n_out):
    if norm_type == "batch":
        return [nn.BatchNorm1d(n_out)]
    elif norm_type == "none":
        return []
    else:
        raise NotImplementedError(f"input norm_type {norm_type} is not implemented")


def __normalization(norm_type, n_out, dim):
    """
        return the normalization layer based on given type and parameter as a list
    """
    if dim == 1:
        return __norm1d(norm_type, n_out)
    elif dim == 2:
        return __norm2d(norm_type, n_out)
    else:
        return NotImplementedError(f"norm generator for dim={dim} is not implemented")


def __activation(activation_type="PReLU", leaky_slope=0):
    """
        return the activation layer based on given type and parameter as a list
    """
    if activation_type == "PReLU":
        return [nn.PReLU()]
    elif activation_type == "ReLU":
        return [nn.ReLU()]
    elif activation_type == "leakyReLU":
        return [nn.LeakyReLU(leaky_slope)]
    elif activation_type == "Tanh":
        return [nn.Tanh()]
    elif activation_type == "Sigmoid":
        return [nn.Sigmoid()]
    elif activation_type == "none":
        return []
    else:
        raise NotImplementedError(f"input activation_type {activation_type} is not implemented")


def __dropout(p_drop, dim):
    if dim == 1:
        if 0 < p_drop < 1:
            return [nn.Dropout(p_drop, inplace=True)]
        elif p_drop == 0:
            return []
        else:
            raise ValueError(f"p_drop should be a value in [0, 1), given: {p_drop}")
    elif dim == 2:
        if 0 < p_drop < 1:
            return [nn.Dropout2d(p_drop, inplace=True)]
        elif p_drop == 0:
            return []
        else:
            raise ValueError(f"p_drop should be a value in [0, 1), given: {p_drop}")
    else:
        return NotImplementedError(f"norm generator for dim={dim} is not implemented")


def Conv2DLayer(c_in, c_out, kernel_size, stride=1, pad=0, norm_type="batch", activation_type="PReLU", leaky_slope=0, p_drop=0, use_sn_init=False):
    """
        create a sequential model based on conv2d layer
        conv -> norm -> activation -> dropout 
    """
    layers = []
    # base conv2d layer
    layers.append(nn.Conv2d(c_in, c_out, kernel_size, stride, pad))
    # spectral norm initialization
    if use_sn_init:
        layers[0] = nn.utils.spectral_norm(layers[0])
    # normalization
    layers.extend(__normalization(norm_type, c_out, dim=2))
    # activarion
    layers.extend(__activation(activation_type, leaky_slope))
    # dropout
    layers.extend(__dropout(p_drop, dim=2))
    return nn.Sequential(*layers)


def LinearLayer(n_in, n_out, norm_type="batch", activation_type="Sigmoid", leaky_slope=0, p_drop=0, bias=True):
    """
        create a sequential model based on linear layer
        linear -> norm -> activation -> dropout 
    """
    layers = []
    # base fc layer
    layers.append(nn.Linear(n_in, n_out, bias))
    # normalization
    layers.extend(__normalization(norm_type, n_out, dim=1))
    # activarion
    layers.extend(__activation(activation_type, leaky_slope))
    # dropout
    layers.extend(__dropout(p_drop, dim=1))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """
        ResBlock SubModule
    """
    def __init__(self, c_in, kernel_size=3, activation_type="ReLU", norm_type="instance",  p_drop=0.5):
        super(ResBlock, self).__init__()
        # check kernel size and set pad size
        assert(kernel_size % 2 == 1)
        n_pad = (kernel_size - 1) // 2
        # init layers
        layers = list()
        layers.append(nn.ReflectionPad2d(n_pad))
        layers.append(Conv2DLayer(c_in, c_in, kernel_size, norm_type="none", activation_type=activation_type, p_drop=p_drop))
        layers.append(nn.ReflectionPad2d(n_pad))
        layers.append(Conv2DLayer(c_in, c_in, kernel_size, norm_type="none", activation_type=activation_type, p_drop=p_drop))
        layers.extend(__norm2d(norm_type, c_in))
        # set main module
        self.res = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.res(x)