from typing import Optional
import os
import pandas as pd
import scipy.io as sio
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import pytorch_lightning as pl
from .utils import transform_rgb


class LitDataModule(pl.LightningDataModule):
    def __init__(self, args):
        self._args = args

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("sequence_data_module")
        parser.add_argument('--data-src', type=str, default=None)
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--num-workers', type=int, default=8)
        parser.add_argument('--sequence-size', type=int, default=32)

        return parent_parser

    def setup(self, stage: Optional[str] = None):
        data_src = self._args.data_src

        if stage == "fit" or stage is None:
            transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.3409, 0.3083,
                        0.2384],  #[0.3392, 0.3069, 0.2372], #[0.3495, 0.3152, 0.2435],
                    std=[0.2436, 0.1835,
                        0.1901]  #[0.2430, 0.1832, 0.1896]  #[0.2429, 0.1824, 0.1904]
                )
            ])

            split_dir = os.path.join(os.path.dirname(__file__), '..', 'tmp')
            self._train_dset = TorchDataset(os.path.join(split_dir, 'train_split.csv'),
                                            os.path.join(data_src, 'images'),
                                            data_src,
                                            sequence_size=self._args.sequence_size,
                                            transform=transform)

            self._val_dset = TorchDataset(os.path.join(split_dir, 'test_split.csv'),
                                          os.path.join(data_src, 'images'),
                                          data_src,
                                          sequence_size=self._args.sequence_size,
                                          transform=transform)

    def train_dataloader(self):
        return DataLoader(self._train_dset,
                          batch_size=self._args.batch_size,
                          num_workers=self._args.num_workers,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self._val_dset,
                          batch_size=self._args.batch_size,
                          num_workers=self._args.num_workers,
                          shuffle=False,
                          pin_memory=True)


class TorchDataset(Dataset):
    #'Characterizes a dataset for PyTorch'
  def __init__(self, csv_file, img_dir, data_dir, sequence_size = 10, transform = None):
    self.img_ids = pd.read_csv(csv_file)
    self.img_dir = img_dir
    self.transform = transform
    self.sequence_size = sequence_size
    
    self.data, self.obs = [], []
    for scene in range(25):
      scene_data, scene_obs = [], []
      for trace in range(5):  #obstacle avoidance
        trace_data, trace_obs = [], []
        for seq in range(39):
          dir = 'scene_' + format(scene, '02d') + '_trace_' + format(trace, '02d') + '_seq_' + format(seq, '02d') + '.mat'
          data = sio.loadmat(data_dir + '/control/' + dir)
          trace_data.append(np.float32(data['data']))
          dir = 'scene_' + format(scene, '02d') + '_trace_' + format(trace, '02d') + '_obs_' + format(seq, '02d') + '.mat'
          obs = sio.loadmat(data_dir + '/control/' + dir)
          rows, columns =  np.float32(data['data']).shape
          trace_obs.append(np.tile(np.float32(obs['obs_active']), (rows, 1)))
        scene_data.append(trace_data)
        scene_obs.append(trace_obs)
      
      trace_data, trace_obs = [], []   #lane tracking
      for seq in range(100):
        dir = 'scene_' + format(scene, '02d') + '_trace_-1' + '_seq_' + format(seq, '02d') + '.mat'
        data = sio.loadmat(data_dir + '/control/' + dir)
        trace_data.append(np.float32(data['data']))
        obs = np.append(np.float32(data['data'])[:,0].reshape(-1,1) - 2, np.ones_like(np.float32(data['data'])[:,1].reshape(-1,1))*8, axis = 1)
        trace_obs.append(obs)
      scene_data.append(trace_data)
      scene_obs.append(trace_obs)
      self.data.append(scene_data)
      self.obs.append(scene_obs)
    
    #for normalizing the output
    self.obs_mean_acc = np.float32(np.array([-0.00140736, -0.00134981]))
    self.obs_std_acc = np.float32(np.array([0.41289764, 0.24781618]))
    self.obs_mean_spd = np.float32(np.array([ 7.81807382, -0.00968291]))
    self.obs_std_spd = np.float32(np.array([0.35971887, 0.10999696]))

    self.track_mean_acc = np.float32(np.array([-0.00593769,  0.0008495]))
    self.track_std_acc = np.float32(np.array([0.20729345, 0.02376962]))
    self.track_mean_spd = np.float32(np.array([ 7.98162067, -0.00958213]))
    self.track_std_spd = np.float32(np.array([0.18146611, 0.03751814]))

  def __len__(self):
    #'Denotes the total number of samples'
    return len(self.img_ids)

  def __getitem__(self, index):
    #'Generates one sample of data'
    # Select sample
    # if self.img_ids.iloc[index, 2] > 0:  #debug for integration
    #   index = index - 1

    scene_id = self.img_ids.iloc[index, 1]  #ids does not change within the sequence
    trace_id = self.img_ids.iloc[index, 2]
    seq_id = self.img_ids.iloc[index, 3]

    if (self.img_ids.iloc[index, 4] < self.sequence_size - 1): #image id in each trace < sequence size - 1
      index = index + self.sequence_size - 1 - self.img_ids.iloc[index, 4]
    images = []
    for i in range(index-self.sequence_size+1, index+1):
      trace_id_tmp = (-1 if trace_id == 5 else trace_id)
      img_name = 'scene_' + format(scene_id, '02d') + '/trace_' + format(trace_id_tmp, '02d') + '/seq_' + format(seq_id, '02d') + '/' + self.img_ids.iloc[i, 0]
      img_path = os.path.join(self.img_dir, img_name)
      image = read_image(img_path) / 255.
      images.append(image)
    
    data_id = self.img_ids.iloc[index-self.sequence_size+1:index+1, 4]   #debug for integration, origin + 1

    state = torch.tensor(self.data[scene_id][trace_id][seq_id][data_id, 0:6])
    obs = torch.tensor(self.obs[scene_id][trace_id][seq_id][data_id, :])
    
    y1 = torch.tensor(self.data[scene_id][trace_id][seq_id][data_id, 3:5])
    y = torch.tensor(self.data[scene_id][trace_id][seq_id][data_id, 6:8])  #3:5 or 6:8
    # y = torch.cat([state[:,1:3], y], dim = 1)
    y = torch.cat([y1, y], dim = 1)


    # import pdb; pdb.set_trace()
    if self.transform:
      X = torch.stack([self.transform(image) for image in images], dim=0)
    
    # state = state[0:-1]  #debug for integration, remove one additional sequence
    # obs = obs[0:-1]
    # y = y[1:]

    return X, state, obs, y
