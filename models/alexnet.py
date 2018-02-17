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

class AlexNet(nn.Module):
  def __init__(self, num_classes):
    super(AlexNet, self).__init__()

    self.features = nn.Sequential(
      nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(256, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )

    self.classifier = nn.Sequential(
      nn.Linear(256 * 6 * 6, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256 * 6 * 6)
    x = self.classifier(x)
    return x


def alexnet(pretrained=False, **kwargs):
  model = AlexNet(**kwargs)
  if pretrained:
    model_path = 'where_it_is'
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['state_dict'])
  return model