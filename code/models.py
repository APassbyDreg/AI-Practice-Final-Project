import random
import torch
import numpy as np
from torch import nn, optim
from tqdm import trange

try:
    from code.model_building_tools import *
    from code.model_sl_tools import *
except:
    from model_building_tools import *
    from model_sl_tools import *


class Estimator(nn.Module):
    """
        estimator
    """
    def __init__(self, input_size, output_size=4):
        super(Estimator, self).__init__()
        self.input_size = input_size
        self.conv1 = Conv2DLayer(6, 16, 5)              # shape = (16, 9, 9)
        self.conv2 = Conv2DLayer(16, 32, 5, pad=1)      # shape = (32, 7, 7)
        self.conv3 = Conv2DLayer(32, 64, 5, pad=1)      # shape = (64, 5, 5)
        self.flatten = nn.Flatten()
        w1, w2 = input_size[-1], input_size[-2]
        n_feat = (w1 - 8) * (w2 - 8) * 64
        self.fc1 = LinearLayer(n_feat, 256, activation_type="PReLU")
        self.fc2 = LinearLayer(256, 256, activation_type="PReLU")
        self.fc3 = LinearLayer(256, output_size, activation_type="none", norm_type="none")

    def forward(self, x: torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        # check input
        assert(len(x.shape) == 4)
        assert(x.shape[1] == self.input_size[1])
        assert(x.shape[2] == self.input_size[2])
        assert(x.shape[3] == self.input_size[3])
        # compute
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def init_model(net, mean=0, std=0.2):
    def func(m):
        try:
            nn.init.normal_(m.weight.data, mean, std)
        except:
            pass
        try:
            nn.init.constant_(m.bias.data, 0.0)
        except:
            pass
    net.apply(func)


class DQN:
    """
        DQN class containing model and training methods
    """
    def __call__(self, x):
        self.model_pred.eval()
        return self.model_pred(x)

    def __init__(self, input_size=[None, 6, 13, 13], output_size=4, lr=5e-4, update_rate=1, gamma=0.9, batch_size=256, batch_num=64) -> None:
        super().__init__()
        self.model_train = Estimator(input_size, output_size)
        self.model_pred = Estimator(input_size, output_size)
        init_model(self.model_train)
        init_model(self.model_pred)
        self.loss = nn.SmoothL1Loss()
        self.optimizer = optim.SGD(self.model_train.parameters(), lr=lr, momentum=0.2)
        self.n_train = 0
        self.update_rate = update_rate
        self.gamma = 0.8
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.lr_shed = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.8)

    def load_checkpoint(self, ckpt_path):
        load_ckpt(self.model_pred, ckpt_path)
        load_ckpt(self.model_train, ckpt_path)

    def train_batch(self, memory):
        loss = 0
        for _ in trange(self.batch_num):
            loss += self.train_once(memory)
        self.lr_shed.step()
        # update pred model
        self.n_train += 1
        if self.n_train % self.update_rate == 0:
            self.model_pred.load_state_dict(self.model_train.state_dict())
            self.model_pred.eval()
        return loss / self.batch_num

    def train_once(self, memory):
        self.model_train.train(True)
        self.optimizer.zero_grad()
        # sample from memory
        samples = random.sample(memory, self.batch_size)
        states_batch, action_batch, reward_batch, next_states_batch, _ = map(np.array, zip(*samples))
        # get target & pred
        q_values_next_target = self.model_pred(next_states_batch).detach().numpy()
        targets_batch = reward_batch + self.gamma * q_values_next_target.max(axis=1)
        targets_batch = torch.FloatTensor(targets_batch.reshape(-1, 1))
        q_values_next_pred = self.model_train(states_batch)
        action_batch = torch.tensor(action_batch.reshape(-1, 1), dtype=torch.int64)
        pred_batch = q_values_next_pred.gather(-1, action_batch)
        # train model
        l = self.loss(pred_batch, targets_batch)
        l.backward()
        self.optimizer.step()
        return l.item()


if __name__ == "__main__":
    # est = Estimator(input_size=[None, 6, 13, 13])
    # print(est)
    x = torch.rand(10, 6, 13, 13)
    # print(est(x))

    dqn = DQN()
    print(dqn(x))

    print(x.dtype)