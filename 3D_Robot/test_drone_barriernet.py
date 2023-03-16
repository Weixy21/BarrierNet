#%%
import torch
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

train_data = sio.loadmat('data/data_train.mat')
train_data = train_data['data']
test_data = sio.loadmat('data/data_imp.mat') 
test_data = test_data['data']

train0 = np.double(train_data[:,0:6])  # x, vx, y, vy, z, vz
train_labels = np.reshape(np.double(train_data[:,6:9]), (len(train_data),3)) #ux, uy, uz
test0 = np.double(test_data[:,0:6]) 
test_labels = np.reshape(np.double(test_data[:,6:9]), (len(test_data),3))
init = test0[0]

mean = np.mean(train0, axis = 0)
std= np.std(train0, axis = 0)


# Initialize the model.
nFeatures, nHidden1, nHidden21, nHidden22, nCls = 6, 512, 128, 128, 3
if barriernet == 1:
    model = models.BarrierNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
    model.load_state_dict(torch.load("model_bn.pth"))
else:
    nFeatures, nHidden1, nHidden21, nHidden22, nCls = 6, 512, 128, 128, 3
    model = models.FCNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
    model.load_state_dict(torch.load("model_fc.pth"))



model.eval()    

px, vx, py, vy, pz, vz = init[0], init[1], init[2], init[3], init[4], init[5]
tr, tr0 = [], 0
ctrl1, ctrl2, ctrl3, safety, ctrl1_real, ctrl2_real, ctrl3_real, p1, p2, loc_x, loc_y, loc_z = [], [], [], [], [], [], [], [], [], [], [], []
dt = 0.1
obs_x, obs_y, obs_z, R = 10, 10, 9, 7
true_x, true_y, true_z = [], [], []

# running on a vehicle
with torch.no_grad():
    for i in range(0,len(test0),1):
        #normalize
        x = (px - mean[0])/std[0]
        v1 = (vx - mean[1])/std[1]
        y = (py - mean[2])/std[2]
        v2 = (vy - mean[3])/std[3]
        z = (pz - mean[4])/std[4]
        v3 = (vz - mean[5])/std[5]
        
        #get safety metric
        safe = (px - obs_x)**4 + (py - obs_y)**4 + (pz - obs_z)**4 - R**4
        safety.append(safe)
        loc_x.append(px)
        loc_y.append(py)
        loc_z.append(pz)
        true_x.append(test0[i][0])
        true_y.append(test0[i][2])
        true_z.append(test0[i][4])

        #prepare for model input
        x_r = Variable(torch.from_numpy(np.array([x,v1,y,v2,z,v3])), requires_grad=False)
        x_r = torch.reshape(x_r, (1,nFeatures))
        x_r = x_r.to(device)
        ctrl = model(x_r, 0)
        
        
        #get the penalty functions
        if barriernet == 1:
            p1.append(model.p1.item()) 
            p2.append(model.p2.item())
        
        #update dynamics/state
        if barriernet == 0:
            ctrl = [ctrl[0,0].item(), ctrl[0,1].item(), ctrl[0,2].item()]
        px = px + vx*dt + 0.5*ctrl[0]*dt**2
        vx = vx + ctrl[0]*dt
        py = py + vy*dt + 0.5*ctrl[1]*dt**2
        vy = vy + ctrl[1]*dt
        pz = pz + vz*dt + 0.5*ctrl[2]*dt**2
        vz = vz + ctrl[2]*dt
               
        ctrl1.append(ctrl[0])
        ctrl2.append(ctrl[1])
        ctrl3.append(ctrl[2])
        ctrl1_real.append(test_labels[i][0])
        ctrl2_real.append(test_labels[i][1])
        ctrl3_real.append(test_labels[i][2])
        tr.append(tr0)
        tr0 = tr0 + dt
print("Implementation done!")


plt.figure(1)
fig, ax = plt.subplots()
plt.plot(tr, ctrl1_real, color = 'red', label = 'Ground truth')
plt.plot(tr, ctrl1, color = 'blue', label = 'BarrierNet')
plt.legend(loc ='upper left', prop={'size': 10})
plt.ylabel('Steering $u_1/(rad/s)$',fontsize=14)
plt.xlabel('time$/s$',fontsize=14)
#plt.ylim(ymin=0.)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.set_rasterized(True)
plt.show()

plt.figure(2)
fig, ax = plt.subplots()
plt.plot(tr, ctrl2_real, color = 'red', label = 'Ground truth')
plt.plot(tr, ctrl2, color = 'blue', label = 'BarrierNet')
plt.legend(prop={'size': 12})
plt.ylabel('Acceleration $u_2/(m/s^2)$',fontsize=14)
plt.xlabel('time$/s$',fontsize=14)
#plt.ylim(ymin=0.)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.set_rasterized(True)
plt.show()

plt.figure(3)
fig, ax = plt.subplots()
plt.plot(tr, ctrl3_real, color = 'red', label = 'Ground truth')
plt.plot(tr, ctrl3, color = 'blue', label = 'BarrierNet')
plt.legend(prop={'size': 12})
plt.ylabel('Acceleration $u_2/(m/s^2)$',fontsize=14)
plt.xlabel('time$/s$',fontsize=14)
#plt.ylim(ymin=0.)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.set_rasterized(True)
plt.show()


fig = plt.figure(4)
ax = plt.axes(projection = '3d')
A, B, C, n = 6, 6, 6, 4
u = np.linspace(-np.pi, np.pi,60)
v = np.linspace(-np.pi/2, np.pi/2, 60)
[u,v] = np.meshgrid(u,v)
x = A*np.sign(np.cos(v))*np.absolute(np.cos(v))**(2/n)*np.sign(np.cos(u))*np.absolute(np.cos(u))**(2/n) + 10
y = B*np.sign(np.cos(v))*np.absolute(np.cos(v))**(2/n)*np.sign(np.sin(u))*np.absolute(np.sin(u))**(2/n) + 10
z = C*np.sign(np.sin(v))*np.absolute(np.sin(v))**(2/n) + 9
ax.plot_surface(x,y,z, cmap = 'viridis', edgecolor = 'none')
ax.plot3D(true_x, true_y, true_z, color = 'red', label = 'Ground truth')
plt.plot(loc_x, loc_y, loc_z, color = 'blue', label = 'BarrierNet')
# plt.plot(loc_x_fc, loc_y_fc, loc_z_fc, linestyle='--', color = 'blue', label = 'FC')
plt.legend(prop={'size': 12}, shadow = False) 
ax.set_ylabel('$y/m$',fontsize=14)
ax.set_xlabel('$x/m$',fontsize=14)
ax.set_zlabel('$z/m$',fontsize=14)
ax.set_box_aspect([20,20,20])
ax.set_rasterized(True)
plt.show()


if barriernet == 1:
    plt.figure(5)
    fig, ax = plt.subplots()
    plt.plot(tr, p1, color = 'green', label = '$p_1(z)$')
    plt.plot(tr, p2, color = 'blue', label = '$p_2(z)$')
    plt.legend(prop={'size': 16})
    plt.ylabel('Penalty functions',fontsize=14)
    plt.xlabel('time$/s$',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_rasterized(True)
    plt.show()


xx = [0,33]
yy = [0,0]
plt.figure(6)
fig, ax = plt.subplots()
plt.plot(xx, yy, linestyle='--', color = 'red', label = 'obstacle boundary')
plt.plot(tr, safety, color = 'blue', label = 'BarrierNet')
# plt.plot(tr, safety_fc, linestyle='--', color = 'blue', label = 'FC')
plt.legend()
plt.ylabel('HOCBF $b(x)$',fontsize=14)
plt.xlabel('time$/s$',fontsize=14)
plt.show()

print("end")


# %%
