import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from qpth.qp import QPFunction
import numpy as np
from my_classes import test_solver as solver


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
        self.obs_x = 40  #obstacle location
        self.obs_y = 15
        self.R = 6   #obstacle size
        self.p1 = 0
        self.p2 = 0
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)
            self.bn22 = nn.BatchNorm1d(nHidden22)

        self.fc1 = nn.Linear(nFeatures, nHidden1).double()
        self.fc21 = nn.Linear(nHidden1, nHidden21).double()
        self.fc22 = nn.Linear(nHidden1, nHidden22).double()
        self.fc31 = nn.Linear(nHidden21, nCls).double()
        self.fc32 = nn.Linear(nHidden22, nCls).double()

        # QP params.
        # from previous layers

    def forward(self, x, sgn):
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
        x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive
        
        # BarrierNet
        x = self.dCBF(x0, x31, x32, sgn, nBatch)
               
        return x

    def dCBF(self, x0, x31, x32, sgn, nBatch):

        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        px = x0[:,0]
        py = x0[:,1]
        theta = x0[:,2]
        v = x0[:,3]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        barrier = (px - self.obs_x)**2 + (py - self.obs_y)**2 - self.R**2
        barrier_dot = 2*(px - self.obs_x)*v*cos_theta + 2*(py - self.obs_y)*v*sin_theta
        Lf2b = 2*v**2
        LgLfbu1 = torch.reshape(-2*(px - self.obs_x)*v*sin_theta + 2*(py - self.obs_y)*v*cos_theta, (nBatch, 1)) 
        LgLfbu2 = torch.reshape(2*(px - self.obs_x)*cos_theta + 2*(py - self.obs_y)*sin_theta, (nBatch, 1))
          
        G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        G = torch.reshape(G, (nBatch, 1, self.nCls)).to(self.device)     
        h = (torch.reshape(Lf2b + (x32[:,0] + x32[:,1])*barrier_dot + (x32[:,0]*x32[:,1])*barrier, (nBatch, 1))).to(self.device) 
        e = Variable(torch.Tensor()).to(self.device)
            
        if self.training or sgn == 1:    
            x = QPFunction(verbose = 0)(Q.double(), x31.double(), G.double(), h.double(), e, e)
        else:
            self.p1 = x32[0,0]
            self.p2 = x32[0,1]
            x = solver(Q[0].double(), x31[0].double(), G[0].double(), h[0].double())
        
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

        self.fc1 = nn.Linear(nFeatures, nHidden1).double()
        self.fc21 = nn.Linear(nHidden1, nHidden21).double()
        self.fc31 = nn.Linear(nHidden21, nCls).double()


    def forward(self, x, sgn):
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
        
        
        