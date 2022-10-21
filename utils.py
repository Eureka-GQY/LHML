import numpy as np
import torch


class DropGrad(torch.nn.Module):
  def __init__(self, rate=0.1, schedule='constant'):
    super(DropGrad, self).__init__()
    self.method = 'gaussian'
    self.rate = rate if self.method != 'gaussian' else np.sqrt(rate/(1 - rate))
    self.schedule = schedule
    self.cur_rate = self.rate

  def update_rate(self, epoch ,stop_epoch):
    if self.schedule == 'constant':
      self.cur_rate = self.rate
    elif self.schedule == 'linear':
      self.cur_rate = self.rate * epoch  / (stop_epoch - 1)

  def forward(self, input):
    output = input * torch.normal(mean=torch.ones_like(input), std=torch.ones_like(input)*self.cur_rate)
    return output