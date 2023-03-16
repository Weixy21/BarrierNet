import torch
import torch.nn as nn
from my_classes import Dataset_list as Dataset
import scipy.io as sio
import models
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
plt.style.use('bmh')

#choose a model
barriernet = 1

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using {} device".format(device))
torch.backends.cudnn.benchmark = True


# Datasets from mat
train_data = sio.loadmat('data/data_train_ocbf.mat')
train_data = train_data['data_train_ocbf']  # data_train for oc controller
test_data = sio.loadmat('data/data_test_ocbf.mat')
test_data = test_data['data_test_ocbf']   # data_test for oc controller
impl_data = sio.loadmat('data/data_ip.mat')
impl_data = impl_data['data_ip']

train0 = np.float32(train_data[:,0:4])
train_labels = np.reshape(np.float32(train_data[:,4]), (len(train_data),1))
test0 = np.float32(test_data[:,0:4])
test_labels = np.reshape(np.float32(test_data[:,4]), (len(test_data),1))
impl0 = np.float32(impl_data)
init = train0[0]

mean = np.mean(train0, axis = 0)
std= np.std(train0, axis = 0)
train0 = (train0 - mean)/std
test0 = (test0 - mean)/std
impl0 = (impl0 - mean)/std

# Initialize the model.
nFeatures, nHidden1, nHidden21, nHidden22, nCls = 4, 72, 24, 24, 1
if barriernet == 1:
    model = models.BarrierNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
    model.load_state_dict(torch.load("model_ocbf_bn.pth"))
else:
    model = models.FCNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
    model.load_state_dict(torch.load("model_ocbf_fc.pth"))
    


    

model.eval()
predic, actual, t = [], [], []
t0 = 0

with torch.no_grad():
    for i in range(len(test0)):
        x, y = Variable(torch.from_numpy(test0[i]), requires_grad=False), test_labels[i]
        x = torch.reshape(x, (1,nFeatures))
        x = x.to(device)
        pred = model(x)
        predic.append(pred.item())
        actual.append(y)
        t.append(t0)
        t0 = t0 + 0.06
print("Test done!")        

pos, vel = init[2], init[3]
tr, tr0 = [], 0
implem, safety, lb, penalty = [], [], [], []
dt = 0.1

# running on a vehicle
with torch.no_grad():
    for i in range(0,len(impl0),10):
        #normalize
        x = (pos - mean[2])/std[2]
        v = (vel - mean[3])/std[3]
        x_ip = impl0[i,0]*std[0] + mean[0] #recover
        #get safety metric
        safe = (x_ip - pos)/vel
        safety.append(safe)
        lb.append(1.8)
        #prepare for model input
        impl0[i,2] = x
        impl0[i,3] = v
        x_r = Variable(torch.from_numpy(impl0[i]), requires_grad=False)
        x_r = torch.reshape(x_r, (1,nFeatures))
        x_r = x_r.to(device)
        ctrl = model(x_r)
        
        if barriernet == 1:
            penalty.append(model.penalty.item())  #only for the barriernet
        
        #update state
        pos = pos + vel*dt + 0.5*ctrl.item()*dt*dt
        vel = vel + ctrl.item()*dt
        
        implem.append(ctrl.item())
        tr.append(tr0)
        tr0 = tr0 + dt
print("Implementation done!")
 

plt.figure(1, figsize=(9,6))
plt.subplot(2, 1, 1)
plt.plot(t, actual, color = 'red', label = 'ground truth')
plt.plot(t, predic, color = 'green', label = 'test')
plt.plot(tr, implem, color = 'blue', label = 'implementation')
plt.legend(prop={'size': 10})
plt.ylabel('Control $u_k/(m/s^2)$',fontsize=14)
plt.xlabel('time$/s$',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

if barriernet == 1:
    plt.subplot(2, 1, 2)
    plt.plot(tr, penalty, color = 'magenta')
    plt.ylabel('Penalty function $p_1(z)$',fontsize=14)
    plt.xlabel('time$/s$',fontsize=14)
    plt.ylim(ymin=0.)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

plt.show()
# if barriernet == 1:
#     plt.savefig('control_ocbf_bn.pdf')
# else:
#     plt.savefig('control_ocbf_fc.pdf')
    