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

train0 = np.float32(train_data[:,0:4])  # x_ip, v_ip, x_i, v_i
train_labels = np.reshape(np.float32(train_data[:,4]), (len(train_data),1))
test0 = np.float32(test_data[:,0:4])
test_labels = np.reshape(np.float32(test_data[:,4]), (len(test_data),1))
impl0 = np.float32(impl_data)
init = train0[0]

# data normalization
mean = np.mean(train0, axis = 0)
std= np.std(train0, axis = 0)
train0 = (train0 - mean)/std
test0 = (test0 - mean)/std
impl0 = (impl0 - mean)/std


# Parameters
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 12}

# Generators
training_set = Dataset(train0, train_labels)
train_dataloader = torch.utils.data.DataLoader(training_set, **params)

test_set = Dataset(test0, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_set, **params)


# Initialize the model.
nFeatures, nHidden1, nHidden21, nHidden22, nCls = 4, 72, 24, 24, 1
if (barriernet == 1):
    model = models.BarrierNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
else:
    model = models.FCNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
print(model)

# Initialize the optimizer.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #Adam
loss_fn = nn.MSELoss()


def train(dataloader, model, loss_fn, optimizer, losses):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 25 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses

def test(dataloader, model, loss_fn, losses):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
    test_loss /= num_batches
    losses.append(test_loss)
    print(f"Test avg loss: {test_loss:>8f} \n")
    return losses

    
epochs = 20 
train_losses, test_losses = [], []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_losses = train(train_dataloader, model, loss_fn, optimizer, train_losses)
    test_losses = test(test_dataloader, model, loss_fn, test_losses)
print("Training Done!")


#save model
if (barriernet == 1):
    torch.save(model.state_dict(), "model_ocbf_bn.pth")
else:
    torch.save(model.state_dict(), "model_ocbf_fc.pth")
print("Saved PyTorch Model State to xx.pth")


# on the test dataset
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
implem, safety, lb = [], [], []
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
        
        #integrate dynamics
        pos = pos + vel*dt + 0.5*ctrl.item()*dt*dt
        vel = vel + ctrl.item()*dt
        
        implem.append(ctrl.item())
        tr.append(tr0)
        tr0 = tr0 + dt
print("Implementation done!")
 

plt.figure(1)
plt.plot(t, predic, color = 'green', label = 'predicted')
plt.plot(t, actual, color = 'red', label = 'actual(optimal)')
plt.plot(tr, implem, color = 'blue', label = 'implemented')
plt.legend()
plt.ylabel('Control')
plt.xlabel('time')
plt.show()
# plt.savefig('control_ocbf_bn.png')

plt.figure(2)    
plt.plot(train_losses, color = 'green', label = 'train')
plt.plot(test_losses, color = 'red', label = 'test')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('time')
plt.ylim(ymin=0.)
plt.show()
# plt.savefig('Loss_ocbf_bn.png')

plt.figure(3)    
plt.plot(tr, safety, color = 'green', label = 'safety')
plt.plot(tr, lb, color = 'red', label = 'lower bound')
plt.legend()
plt.ylabel('Safety')
plt.xlabel('time')
plt.show()
# plt.savefig('Safety_ocbf_bn.png')

print("end")