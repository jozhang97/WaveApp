from __future__ import division

import os, sys, pdb, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchaudio.datasets as dset
import torchaudio.transforms as transforms
from data.arctic import Arctic
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models
from models.alexnet import AlexNet
from tensorboardX import SummaryWriter
from scipy.stats import entropy
from math import log, exp

import numpy as np

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))
# TODO Fix ^
model_names = ['alexnet']

writer = SummaryWriter()

parser = argparse.ArgumentParser(description='Trains AlexNet on CMU Arctic', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='data', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='arctic', choices=['arctic', 'vctk', 'yesno'])
parser.add_argument('--arch', metavar='ARCH', default='alexnet', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./logs', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='./logs', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
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

def main():
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

  # Data loading code
  # Any other preprocessings? http://pytorch.org/audio/transforms.html
  sample_length = 10000
  scale = transforms.Scale()
  padtrim = transforms.PadTrim(sample_length)
  downmix = transforms.DownmixMono()
  transforms_audio = transforms.Compose([
    scale, padtrim, downmix
  ])

  if not os.path.isdir(args.data_path):
    os.makedirs(args.data_path)
  train_dir = os.path.join(args.data_path, 'train')
  val_dir = os.path.join(args.data_path, 'val')

  #Choose dataset to use
  if args.dataset == 'arctic':
    # TODO No ImageFolder equivalent for audio. Need to create a Dataset manually
    train_dataset = Arctic(train_dir, transform=transforms_audio, download=True)
    val_dataset = Arctic(val_dir, transform=transforms_audio, download=True)
    num_classes = 4
  elif args.dataset == 'vctk':
    train_dataset = dset.VCTK(train_dir, transform=transforms_audio, download=True)
    val_dataset = dset.VCTK(val_dir, transform=transforms_audio, download=True)
    num_classes = 10
  elif args.dataset == 'yesno':
    train_dataset = dset.YESNO(train_dir, transform=transforms_audio, download=True)
    val_dataset = dset.YESNO(val_dir, transform=transforms_audio, download=True)
    num_classes = 2
  else:
    assert False, 'Dataset is incorrect'

  train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    # pin_memory=True, # What is this?
    # sampler=None     # What is this?
  )
  val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)


  #Feed in respective model file to pass into model (alexnet.py)
  print_log("=> creating model '{}'".format(args.arch), log)
  # Init model, criterion, and optimizer
  # net = models.__dict__[args.arch](num_classes)
  net = AlexNet(num_classes)
  #
  print_log("=> network :\n {}".format(net), log)

  # net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

  # define loss function (criterion) and optimizer
  criterion = torch.nn.CrossEntropyLoss()

  # Define stochastic gradient descent as optimizer (run backprop on random small batch)
  optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=True)

  #Sets use for GPU if available
  if args.use_cuda:
    net.cuda()
    criterion.cuda()

  recorder = RecorderMeter(args.epochs)
  # optionally resume from a checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print_log("=> loading checkpoint '{}'".format(args.resume), log)
      checkpoint = torch.load(args.resume)
      recorder = checkpoint['recorder']
      args.start_epoch = checkpoint['epoch']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
    else:
      print_log("=> no checkpoint found at '{}'".format(args.resume), log)
  else:
    print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

  if args.evaluate:
    validate(val_loader, net, criterion, 0, log, val_dataset)
    return

  # Main loop
  start_time = time.time()
  epoch_time = AverageMeter()

  # Training occurs here
  for epoch in range(args.start_epoch, args.epochs):
    current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

    need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
    need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

    print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

    print("One epoch")
    # train for one epoch
    # Call to train (note that our previous net is passed into the model argument)
    train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, log, train_dataset)

    # evaluate on validation set
    #val_acc,   val_los   = extract_features(test_loader, net, criterion, log)
    val_acc,   val_los   = validate(val_loader, net, criterion, epoch, log, val_dataset)
    is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc)

    save_checkpoint({
      'epoch': epoch + 1,
      'arch': args.arch,
      'state_dict': net.state_dict(),
      'recorder': recorder,
      'optimizer' : optimizer.state_dict(),
    }, is_best, args.save_path, 'checkpoint.pth.tar')

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()
    recorder.plot_curve( os.path.join(args.save_path, 'curve.png') )

  log.close()

# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log, train_dataset):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  # switch to train mode
  model.train()

  end = time.time()
  for i, (input, target) in enumerate(train_loader):
    # TODO Do this in dataset specific file, this is just to get it to run.
    #target = np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 0])  # Just for debugging yesno

    #import ipdb; ipdb.set_trace()
    input = train_dataset.preprocess_input(input)
    target = train_dataset.preprocess_input(target)
    target = target.type(torch.LongTensor)
    try:
      target = torch.from_numpy(target)
    except RuntimeError:
      pass

    input = input.view(input.size(0), 1, input.size(1))  # Flip channels and length

    # measure data loading time
    data_time.update(time.time() - end)

    if args.use_cuda:
      target = target.cuda(async=True)
      input = input.cuda()
      
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)
    output = train_dataset.postprocess_target(output)
    #import ipdb; ipdb.set_trace()
    loss = criterion(output, target_var)

    entropy_val = 0
    perplexity_val = 0
    for batch_output in output:
      entropy_val += get_entropy(softmax(batch_output.data.cpu().numpy()))
      perplexity_val += get_perplexity(entropy_val)

    # measure accuracy and record loss
    #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    prec1, prec5 = accuracy(output.data, target, topk=(1, 2))  # we don't have 5 classes yet lol 
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))

    if i == 0:
      writer.add_scalar('Training Loss', loss.data[0], epoch)
      writer.add_scalar('Accuracy Top 1', prec1[0], epoch)
      writer.add_scalar('Accuracy Top 5', prec5[0], epoch)
      writer.add_scalar('Entropy', entropy_val, epoch)
      writer.add_scalar('Perplexity', perplexity_val, epoch)
      import ipdb; ipdb.set_trace(context=21)
      for input_index in range(len(input)):
        writer.add_audio("First Audio Per Epoch Sample " + str(input_index),
                                     input[input_index], sample_rate=16000)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % args.print_freq == 0:
      #import ipdb; ipdb.set_trace()
      print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
            'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
            'Loss {loss.val:.4f} ({loss.avg:.4f})   '
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
  print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
  return top1.avg, losses.avg

def validate(val_loader, model, criterion, epoch, log, val_dataset):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    input = val_dataset.preprocess_input(input)
    target = val_dataset.preprocess_input(target)
    target = target.type(torch.LongTensor)

    input = input.view(input.size(0), 1, input.size(1))  # Flip channels and leng

    if args.use_cuda:
      target = target.cuda(async=True)
      input = input.cuda()

    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output = model(input_var)
    ouptut = val_dataset.postprocess_target(output)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))
    if i == 0:
      writer.add_scalar('Validation Loss', loss.data[0], epoch)
      writer.add_scalar('Validation Accuracy Top 1', prec1[0], epoch)
      writer.add_scalar('Validation Accuracy Top 5', prec5[0], epoch)

  print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

  return top1.avg, losses.avg

def extract_features(val_loader, model, criterion, log):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    if args.use_cuda:
      target = target.cuda(async=True)
      input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output, features = model([input_var])


    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))

  print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

  return top1.avg, losses.avg

def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()

def save_checkpoint(state, is_best, save_path, filename):
  filename = os.path.join(save_path, filename)
  torch.save(state, filename)
  if is_best:
    bestname = os.path.join(save_path, 'model_best.pth.tar')
    shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.learning_rate
  assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
  for (gamma, step) in zip(gammas, schedule):
    if (epoch >= step):
      lr = lr * gamma
    else:
      break
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr

def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

def get_entropy(label_probs):
  return -sum([((i+1e-10) * log((i+1e-10), 2)) for i in label_probs])

def get_perplexity(entropy):
  return exp(entropy)

def softmax(x):
    return np.exp(x+1e-10) / np.sum(np.exp(x+1e-10), axis=0)

if __name__ == '__main__':
  main()
