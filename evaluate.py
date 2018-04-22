from __future__ import division

import os, sys, pdb, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchaudio.datasets as dset
import torchaudio.transforms as transforms
from utils import AverageMeter
import models
from models.alexnet import AlexNet
from math import log, exp

from scipy.io import wavfile
import numpy as np
from torch.autograd import Variable

model_names = ['alexnet']

parser = argparse.ArgumentParser(description='Trains AlexNet on CMU Arctic', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--arch', metavar='ARCH', default='alexnet', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
# Checkpoints
parser.add_argument('--save_path', type=str, default='./logs', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='trained.tar', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--file_name', type=str, default="test.wav", help='Run evaluation on this file.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
args = parser.parse_args()

# Use GPU if available
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

if args.manualSeed is None:
  args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
  torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

def evaluate():
  num_classes = 4

  # Init logger
  if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
  log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
  print_log('save path : {}'.format(args.save_path), log)
  state = {k: v for k, v in args._get_kwargs()}
  print_log(state, log)
  print_log("Random Seed: {}".format(args.manualSeed), log)
  print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
  print_log("torch  version : {}".format(torch.__version__), log)
  print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

  # Any other preprocessings? http://pytorch.org/audio/transforms.html
  sample_length = 10000
  scale = transforms.Scale()
  padtrim = transforms.PadTrim(sample_length)
  transforms_audio = transforms.Compose([
    scale, padtrim
  ])


  # Data loading
  fs, data = wavfile.read(args.file_name)
  data = torch.from_numpy(data).float()
  data = data.unsqueeze(1)
  audio = transforms_audio(data)
  audio = Variable(audio)
  audio = audio.view(1, -1)
  audio = audio.unsqueeze(0)


  #Feed in respective model file to pass into model (alexnet.py)
  print_log("=> creating model '{}'".format(args.arch), log)

  # Init model, criterion, and optimizer
  # net = models.__dict__[args.arch](num_classes)
  net = AlexNet(num_classes)
  print_log("=> network :\n {}".format(net), log)


  #Sets use for GPU if available
  if args.use_cuda:
    net.cuda()

  # optionally resume from a checkpoint
  # Need same python version that the resume was in
  if args.resume:
    if os.path.isfile(args.resume):
      print_log("=> loading checkpoint '{}'".format(args.resume), log)
      if args.ngpu == 0:
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
      else:
        checkpoint = torch.load(args.resume)

      recorder = checkpoint['recorder']
      args.start_epoch = checkpoint['epoch']
      net.load_state_dict(checkpoint['state_dict'])
      print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
    else:
      print_log("=> no checkpoint found at '{}'".format(args.resume), log)
  else:
    print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

  net.eval()
  if args.use_cuda:
    audio = audio.cuda()
  output = net(audio)
  print(output)
  # TODO postprocess output to a string representing the person speaking
  # ouptut = val_dataset.postprocess_target(output)
  return


def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()


if __name__ == '__main__':
  evaluate()
