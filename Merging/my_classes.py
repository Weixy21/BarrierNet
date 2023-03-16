import torch

class Dataset(torch.utils.data.Dataset):
  #'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
    #'Initialization'
    self.labels = labels
    self.list_IDs = list_IDs

  def __len__(self):
    #'Denotes the total number of samples'
    return len(self.list_IDs)

  def __getitem__(self, index):
    #'Generates one sample of data'
    # Select sample
    ID = self.list_IDs[index]

    # Load data and get label
    X = torch.load('data/' + ID + '.pt')
    y = self.labels[ID]

    return X, y

class Dataset_list(torch.utils.data.Dataset):
  #'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels):
    #'Initialization'
    self.labels = labels
    self.features = features

  def __len__(self):
    #'Denotes the total number of samples'
    return len(self.features)

  def __getitem__(self, index):
    #'Generates one sample of data'

    # Load data and get label
    X = self.features[index]
    y = self.labels[index]

    return X, y


