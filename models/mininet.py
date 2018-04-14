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

class MiniNet(nn.Module):
  def __init__(self, num_classes):
    super(MiniNet, self).__init__()

    self.latent_vector_size = 2560

    self.classifier = nn.Sequential(
      nn.Linear(self.latent_vector_size, 1024),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(1024, 256),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(256, num_classes),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    return self.classifier(x)


def mininet(pretrained=False, **kwargs):
  model = MiniNet(**kwargs)
  if pretrained:
    model_path = './logs'
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['state_dict'])
  return model
