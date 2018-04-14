import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



def conv3x3(in_channels, out_channels, stride=1):
    """
    A simple 3x3 (kernel size) convolution layer
    """
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class SimpleBlock(nn.Module):
    """
    An implementation of a block that keeps same size throughout, simple runs
    a conv/relu -> conv/downsample/relu with same size (no expansion)
    """
    expansion_factor = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        # add batch norm in between?
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels, out_channels, stride)
        # add batch norm again?

        self.conv_layers = nn.Sequential(self.conv1, self.relu, self.conv2)

        self.downsample = downsample
        self.stride = stride


      def forward(self, x):
          residual = x

          x = self.conv_layers(x)

          if self.downsample not None:
            residual = self.downsample(residual)

          x += residual
          x = self.relu(x)

          return x



class ResNet(nn.Module):

    def __init__(self, block, layer_sizes, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # create block classes (which are nn.Module, can be called like x = block(x))
        """
        block classes implement different downsampling, as well as block structure. for example, 
        having conv, relu, conv, downsample + add residual, or more convs in between, etc.

        """

        # initial 7x7 conv layer
        self.conv1 = nn.Conv1d(1, 96, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # have an initial max/avg pool 

        self.layer1 = self.generate_layer(block, 64, layer_sizes[0])
        self.layer2 = self.generate_layer(block, 128, layer_sizes[1], stride=2)
        self.layer3 = self.generate_layer(block, 256, layer_sizes[2], stride=2)

        # CAN CHANGE PARAMETERS FOR EACH Conv1D/ReLU LAYER

        # figure out what is the groups parameter in nn.Conv1D
        # initial_avg_pool = nn.AvgPool1D(kernel_size=3, stride=2)
        # block1 = generate_block(1, 96, kernel_size=11, stride=4, padding=0, dilation=2, pool=initial_avg_pool)
        # block2 = generate_block(96, 256, kernel_size=5, stride=2, padding=2, dilation=4)
        # block3 = generate_block(256, 384, kernel_size=3, stride=2, padding=1, dilation=8)
        # block4 = generate_block(384, 384, kernel_size=3, stride=1, padding=1, groups=2, dilation=16)
        # block5 = generate_block(384, 256, kernel_size=3, stride=1, padding=1, groups=2, dilation=32)
        # block6 = generate_block(256, 96, kernel_size=3, stride=1, padding=1, groups=2, dilation=64)

        # self.blocks = [block1, block2, block3, block4, block5, block6]

        self.latent_vector_size = 256 * 27 # change this potentially

        self.classifier = nn.Sequential(
          nn.Linear(self.latent_vector_size, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(4096, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(4096, num_classes),
          nn.Softmax(dim=1)
        )

    def generate_layer(self, block, channels, num_blocks, stride=1):
        downsample = None
        out_channels = channels * block.expansion_factor
        # if our stride is greater than 1 or we see the input channels is not same as output channels,
        # we need to define a downsample function in order to properly add residuals
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        blocks = []
        # add initial block (which could need downsampling)
        blocks.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, num_blocks):
            blocks.append(block(self.in_channels, channels))

        return nn.Sequential(*blocks)



    # def generate_block(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, pool=None):
    #     """
    #     Generate a block part of our ResNet. 
    #     A block contains a Conv1d layer with specified parameters, a ReLU, and an optional pooling layer.
    #     """
    #     features = []
    #     features.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    #     features.append(nn.ReLU(inplace=True))
    #     if pool:
    #         features.append(pool)
    #     return nn.Sequential(*features)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # max/avg pool

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # avg pool again?
        # classify layer
        x = self.classifier(x)

        return x


def resnet(pretrained=False, **kwargs):
      model = ResNet(block=BasicBlock, layer_sizes=[2,2,2,2], **kwargs)
      if pretrained:
        model_path = './logs'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
      return model
