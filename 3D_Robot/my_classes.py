import torch
from cvxopt import solvers, matrix
import numpy as np

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
  def __init__(self, features, labels, mode='train'):
    #'Initialization'
    self.labels = labels
    self.features = features
    self.mode = mode

  def __len__(self):
    #'Denotes the total number of samples'
    return len(self.features)

  def __getitem__(self, index):
    #'Generates one sample of data'
    
    if self.mode == 'test':
        index = 0

    # Load data and get label
    X = self.features[index]
    y = self.labels[index]

    return X, y

def test_solver(Q, p, G, h):
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())
    
    solvers.options['show_progress'] = False
    
    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)
    
    
    return sol['x']
