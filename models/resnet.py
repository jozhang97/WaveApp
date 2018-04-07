import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        # CAN CHANGE PARAMETERS FOR EACH Conv1D/ReLU LAYER

        # figure out what is the groups parameter in nn.Conv1D
        initial_avg_pool = nn.AvgPool1D(kernel_size=3, stride=2)
        block1 = generate_block(1, 96, kernel_size=11, stride=4, padding=0, dilation=2, pool=initial_avg_pool)
        block2 = generate_block(96, 256, kernel_size=5, stride=2, padding=2, dilation=4)
        block3 = generate_block(256, 384, kernel_size=3, stride=2, padding=1, dilation=8)
        block4 = generate_block(384, 384, kernel_size=3, stride=1, padding=1, groups=2, dilation=16)
        block5 = generate_block(384, 256, kernel_size=3, stride=1, padding=1, groups=2, dilation=32)
        block6 = generate_block(256, 96, kernel_size=3, stride=1, padding=1, groups=2, dilation=64)

        self.blocks = [block1, block2, block3, block4, block5, block6]

        self.latent_vector_size = 256 * 27 # change this potentially

        self.classifier = nn.Sequential(
          nn.Linear(self.latent_vector_size, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(4096, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(4096, num_classes),
        )


    def generate_block(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, pool=None):
        """
        Generate a block part of our ResNet. 
        A block contains a Conv1d layer with specified parameters, a ReLU, and an optional pooling layer.
        """
        features = []
        features.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
        features.append(nn.ReLU(inplace=True))
        if pool:
            features.append(pool)
        return nn.Sequential(*features)


    def forward(self, x):
        # temp = 0
        # for i in range(len(self.sequentials)):
        #     if i == 0:
        #         temp = self.sequentials[i](x)
        #     else:
        #         temp = self.sequentials[i](x) + temp

        # return temp


def resnet(pretrained=False, **kwargs):
      model = ResNet(**kwargs)
      if pretrained:
        model_path = './logs'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
      return model
