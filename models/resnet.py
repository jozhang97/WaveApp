import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        # You can first construct a python list of nn.Modules and unpack it into a nn.Sequential
        feats = []
        self.sequentials = []
        for i in range(10):
            feats.append(nn.Conv1d(1, 96, 11, stride=4, padding=0, dilation=2))
            feats.append(nn.MaxPool1d(kernel_size=3, stride=2))
            feats.append(nn.Conv1d(96, 256, kernel_size=5, stride=2, padding=2, groups=2, dilation=4))
            feats.append(nn.MaxPool1d(kernel_size=3, stride=2))
            #Pass up
            # if i == 0:
            #     sequentials.append(nn.Sequential(*feats))
            # else:
            #     sequentials.append(nn.Sequential(*feats) + sequentials[i-1])
            self.sequentials.append(nn.Sequential(*feats))



        self.features = nn.Sequential(*feats)

        # self.features = nn.Sequential(
        #   nn.Conv1d(1, 96, 11, stride=4, padding=0, dilation=2),
        #   nn.MaxPool1d(kernel_size=3, stride=2),
        #   nn.Conv1d(96, 256, kernel_size=5, stride=2, padding=2, groups=2, dilation=4),
        #   nn.MaxPool1d(kernel_size=3, stride=2),
        #
        #   nn.Conv1d(256, 384, kernel_size=3, stride=2, padding=1, dilation=8),
        #   nn.MaxPool1d(kernel_size=3, stride=2),
        #   nn.Conv1d(384, 384, kernel_size=3, stride=1, padding=1, groups=2, dilation=16),
        #   nn.MaxPool1d(kernel_size=3, stride=2),
        #
        #   nn.Conv1d(384, 256, kernel_size=3, stride=1, padding=1, groups=2, dilation=32),
        #   nn.MaxPool1d(kernel_size=3, stride=2),
        #   nn.Conv1d(384, 256, kernel_size=3, stride=1, padding=1, groups=2, dilation=32),
        #   nn.MaxPool1d(kernel_size=3, stride=2),
        # )


    def forward(self, x):
        temp = 0
        for i in range(0, len(self.sequentials)):
            if i == 0:
                temp = self.sequentials[i](x)
            else:
                temp = self.sequentials[i](x) + temp

        return temp


def resnet(pretrained=False, **kwargs):
      model = ResNet(**kwargs)
      if pretrained:
        model_path = './logs'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
      return model
