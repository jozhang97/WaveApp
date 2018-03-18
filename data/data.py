import torch.utils.data as data

class DataProperty(data.Dataset):
  def preprocess_input(self, input):
  	return input

  def preprocess_target(self, target):
    return target

  def postprocess_target(self, target):
    return target