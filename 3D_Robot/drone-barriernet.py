import torch
import torch.nn as nn
from my_classes import Dataset_list as Dataset
import scipy.io as sio
import models
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
plt.style.use('bmh')

#choose a barriernet or not
barriernet = 1

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using {} device".format(device))
torch.backends.cudnn.benchmark = True


# Datasets
train_data = sio.loadmat('data/data_train.mat')
train_data = train_data['data']
test_data = sio.loadmat('data/data_test.mat') 
test_data = test_data['data']

train0 = np.double(train_data[:,0:6])  # x, vx, y, vy, z, vz
train_labels = np.reshape(np.double(train_data[:,6:9]), (len(train_data),3)) #ux, uy, uz
test0 = np.double(test_data[:,0:6]) 
test_labels = np.reshape(np.double(test_data[:,6:9]), (len(test_data),3))
init = test0[0]

mean = np.mean(train0, axis = 0)
std= np.std(train0, axis = 0)
train0 = (train0 - mean)/std
test0 = (test0 - mean)/std


# Parameters
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 20}

# Generators
training_set = Dataset(train0, train_labels)
train_dataloader = torch.utils.data.DataLoader(training_set, **params)

test_set = Dataset(test0, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_set, **params)


# Initialize the model.
nFeatures, nHidden1, nHidden21, nHidden22, nCls = 6, 512, 128, 128, 3
if barriernet == 1:
    model = models.BarrierNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
else:
    nFeatures, nHidden1, nHidden21, nHidden22, nCls = 6, 512, 128, 128, 3 
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
        pred = model(X, 1)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 15 == 0:  
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
            pred = model(X, 1)
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
if barriernet == 1:
    torch.save(model.state_dict(), "model_bn.pth")
else:
    torch.save(model.state_dict(), "model_fc.pth")
print("Saved PyTorch Model State to model_xx.pth")




model.eval()    
tr = []
ctrl1, ctrl2, ctrl3, ctrl1_real, ctrl2_real, ctrl3_real = [], [], [], [], [], []
t0 = 0

with torch.no_grad():
    for i in range(0,len(train0),1):
        x = Variable(torch.from_numpy(train0[i]), requires_grad=False)
        x = torch.reshape(x, (1,nFeatures))
        x = x.to(device)
        #print(x)
        ctrl = model(x, 0)
        
        if barriernet == 1:
            ctrl1.append(ctrl[0])
            ctrl2.append(ctrl[1])
            ctrl3.append(ctrl[2])
        else:
            ctrl1.append(ctrl[0,0].item())
            ctrl2.append(ctrl[0,1].item())
            ctrl3.append(ctrl[0,2].item())
        ctrl1_real.append(train_labels[i][0])
        ctrl2_real.append(train_labels[i][1])
        ctrl3_real.append(train_labels[i][2])
        tr.append(t0)
        t0 = t0 + 0.1

print("Test done!")    


plt.figure(1)
plt.plot(tr, ctrl1_real, color = 'red', label = 'ground truth, $u_1$')
plt.plot(tr, ctrl1, linestyle='--', color = 'red', label = 'test, $u_1$')
plt.plot(tr, ctrl2_real, color = 'green', label = 'ground truth, $u_2$')
plt.plot(tr, ctrl2, linestyle='--', color = 'green', label = 'test, $u_2$')
plt.plot(tr, ctrl3_real, color = 'blue', label = 'ground truth, $u_3$')
plt.plot(tr, ctrl3, linestyle='--', color = 'blue', label = 'test, $u_3$')
plt.legend()
plt.ylabel('Accelerations (controls)')
plt.xlabel('time')
plt.show()
                      


plt.figure(2)    
plt.plot(train_losses, color = 'green', label = 'train')
plt.plot(test_losses, color = 'red', label = 'test')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('time')
plt.ylim(ymin=0.)
    
plt.show()    
    
    
print("end")
