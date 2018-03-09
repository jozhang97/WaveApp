import torch
import torchaudio.datasets as dset
from torchaudio import transforms 

transform = transforms.Compose([
        transforms.Scale(), 
        transforms.PadTrim(100000)
        ])

train_dataset = dset.YESNO("data", transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(
  train_dataset,
  batch_size=10,
  )

for i, (input, target) in enumerate(train_loader):
  import ipdb; ipdb.set_trace(context=21)
  print("HI")





""" Vision MNIST test"""
"""
import torchvision.datasets as vdset
from torchvision import transforms as vtransforms

transform = vtransforms.Compose([
        vtransforms.ToTensor()
        ])

mnist = vdset.MNIST("data", transform=transform, download=True)

mnist_loader = torch.utils.data.DataLoader(
  mnist,
  batch_size=10,
  num_workers=1)

for id, (data, target) in enumerate(mnist_loader):
  pass
"""

