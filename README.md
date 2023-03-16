# BarrierNet

A safety guaranteed neural network controller for autonomous systems 

There are three simple control demos (traffic merging, 2D and 3D robot control) and one vision-based end-to-end autonomous driving demo.

## Setup

    ```
    $ conda create -n bnet python=3.8
    $ conda activate bnet
    $ pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    $ pip install pytorch-lightning==1.5.8 opencv-python==4.5.2.54 matplotlib==3.5.1 ffio==0.1.0  descartes==1.1.0  pyrender==0.1.45  pandas==1.3.5 shapely==1.7.1 scikit-video==1.1.11 scipy==1.6.3 h5py==3.1.0
    $ pip install qpth cvxpy cvxopt
    ```
Install `vista`.
```
$ conda activate bnet
$ cd vista
$ pip install -e .
```
