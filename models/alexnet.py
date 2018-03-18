import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

'''
Based off of https://github.com/jiecaoyu/pytorch_imagenet
Good place to start for ResNet:
https://github.com/D-X-Y/ResNeXt-DenseNet/blob/master/models/resnet.py
has a block class
has a wrapper class that merges blocks together
has functions that output the end product
'''

# Input must be N x C x L
# (batch_size x num_channels x length)

class AlexNet(nn.Module):
  def __init__(self, num_classes):
    super(AlexNet, self).__init__()

    self.features = nn.Sequential(
      nn.Conv1d(1, 96, 11, stride=4, padding=0, dilation=2),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=3, stride=2),
      nn.Conv1d(96, 256, kernel_size=5, stride=2, padding=2, groups=2, dilation=4),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=3, stride=2),
      nn.Conv1d(256, 384, kernel_size=3, stride=2, padding=1, dilation=8),
      nn.ReLU(inplace=True),
      nn.Conv1d(384, 384, kernel_size=3, stride=1, padding=1, groups=2, dilation=16),
      nn.ReLU(inplace=True),
      nn.Conv1d(384, 256, kernel_size=3, stride=1, padding=1, groups=2, dilation=32),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=3, stride=2),
    )

    self.latent_vector_size = 256 * 27

    self.classifier = nn.Sequential(
      nn.Linear(self.latent_vector_size, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), self.latent_vector_size)
    x = self.classifier(x)
    return x


def alexnet(pretrained=False, **kwargs):
  model = AlexNet(**kwargs)
  if pretrained:
    model_path = './logs'
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['state_dict'])
  return model
