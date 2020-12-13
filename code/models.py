import torch
from torch import nn

try:
    from code.model_building_tools import *
except:
    from model_building_tools import *



class Estimator(nn.Module):
    """
        estimator
    """
    def __init__(self, input_size=[None, 4, 9, 9], output_size=4):
        super(Estimator, self).__init__()
        self.input_size = input_size
        self.conv1 = Conv2DLayer(4, 8, 5, pad=1)    # shape = (8, 7, 7)
        self.conv2 = Conv2DLayer(8, 16, 5, pad=1)   # shape = (16, 5, 5)
        self.conv3 = Conv2DLayer(16, 32, 3)         # shape = (32, 3, 3)
        self.flatten = nn.Flatten()
        w1, w2 = input_size[-1], input_size[-2]
        n_feat = (w1 - 6) * (w2 - 6) * 32
        self.fc1 = LinearLayer(n_feat, 512, activation_type="PReLU")
        self.fc2 = LinearLayer(512, 128, activation_type="PReLU")
        self.fc3 = LinearLayer(128, output_size, activation_type="none", norm_type="none")
        self.out_activation = nn.Softmax()

    def forward(self, x: torch.Tensor):
        # check input
        assert(len(x.shape) == 4)
        assert(x.shape[1] == self.input_size[1])
        assert(x.shape[2] == self.input_size[2])
        assert(x.shape[3] == self.input_size[3])
        # compute
        x = self.expand(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.out_activation(x)


if __name__ == "__main__":
    est = Estimator()
    print(est)
    x = torch.rand(4, 1, 9, 9)
    print(est(x))