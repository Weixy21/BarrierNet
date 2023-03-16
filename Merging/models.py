import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from qpth.qp import QPFunction


class BarrierNet(nn.Module):
    def __init__(self, nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.bn = bn
        self.nCls = nCls
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.device = device
        self.penalty = 0
        
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)
            self.bn22 = nn.BatchNorm1d(nHidden22)

        self.fc1 = nn.Linear(nFeatures, nHidden1)
        self.fc21 = nn.Linear(nHidden1, nHidden21)
        self.fc22 = nn.Linear(nHidden1, nHidden22)
        self.fc31 = nn.Linear(nHidden21, nCls)
        self.fc32 = nn.Linear(nHidden22, nCls)

        # QP params.
        # from previous layers

    def forward(self, x):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.std + self.mean
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        
        x21 = F.relu(self.fc21(x))
        if self.bn:
            x21 = self.bn21(x21)
        x22 = F.relu(self.fc22(x))
        if self.bn:
            x22 = self.bn22(x22)
        
        x31 = self.fc31(x21)
        x32 = self.fc32(x22)
        x32 = 4*nn.Sigmoid()(x32)  # positive requirement for the CBF param.
        
        #record the penalty function
        self.penalty = x32.cpu()

        #BarrierNet
        x = self.dCBF(x0, x31, x32, nBatch) 
        return x


    def dCBF(self, x0, x31, x32, nBatch):

        # Set up the qp
        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        G = Variable(1.8*torch.eye(self.nCls))
        G = G.unsqueeze(0).expand(nBatch, 1, self.nCls).to(self.device)
        h = (torch.reshape(x0[:,1] - x0[:,3], (nBatch, self.nCls)) + x32*torch.reshape((x0[:,0] - x0[:,2] - 1.8*x0[:,3]), (nBatch, self.nCls))).to(self.device)
        e = Variable(torch.Tensor()).to(self.device)
         
        x = QPFunction(verbose=0)(Q, x31, G, h, e, e) 

        return x


    
    
class FCNet(nn.Module):
    def __init__(self, nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.nCls = nCls
        self.mean = mean
        self.std = std
        self.device = device
        self.bn = bn
        
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)

        self.fc1 = nn.Linear(nFeatures, nHidden1)
        self.fc21 = nn.Linear(nHidden1, nHidden21)
        self.fc31 = nn.Linear(nHidden21, nCls)


    def forward(self, x):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        
        x21 = F.relu(self.fc21(x))
        if self.bn:
            x21 = self.bn21(x21)
        
        x31 = self.fc31(x21)
        
        return x31