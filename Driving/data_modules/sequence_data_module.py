from typing import Optional
import os
import pandas as pd
from scipy.io import loadmat
import torch
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
        parser.add_argument('--data-tgt', type=str, default=None)
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--num-workers', type=int, default=8)
        parser.add_argument('--sequence-size', type=int, default=32)
        parser.add_argument('--regenerate-split', action='store_true')
        parser.add_argument('--use-fixed-standardize', action='store_true')
        parser.add_argument('--non-obstacle-ds', type=float, default=-2)
        parser.add_argument('--use-color-jitter', action='store_true')
        parser.add_argument('--use-old-indexing', action='store_true')
        parser.add_argument('--exclude-lf-data', action='store_true')

        return parent_parser

    def prepare_data(self):
        # copy and unpack data if necessary
        if self._args.data_tgt is not None:
            raise NotImplementedError

        # if no train/val/test split, generate ones
        data_src = self._args.data_src
        meta_data = {
            'scene': [],
            'trace': [],
            'sequence': [],
            'frame': [],
            'image': [],
            'control': [],
            'obstacle': [],
            'global_sequence_id': [],
        }
        img_root_dir = os.path.join(data_src, 'images')
        scenes = sorted(os.listdir(img_root_dir))
        global_seq_id = 0
        for scene in scenes:
            traces = sorted(os.listdir(os.path.join(img_root_dir, scene)))
            for trace in traces:
                seqs = sorted(os.listdir(os.path.join(img_root_dir, scene, trace)))
                for seq in seqs:
                    ctrl_path = os.path.join(data_src,
                        'control', '_'.join([scene, trace, seq]) + '.mat')
                    if int(trace.split('_')[-1]) != -1:
                        obs_path = os.path.join(data_src, 'control',
                            '_'.join([scene, trace, seq.replace('seq', 'obs')]) + '.mat')
                    else:
                        obs_path = 'None'
                    seq_dir = os.path.join(img_root_dir, scene, trace, seq)
                    imgs = sorted(os.listdir(seq_dir))
                    for img in imgs:
                        img_path = os.path.join(seq_dir, img)
                        meta_data['scene'].append(int(scene.split('_')[-1]))
                        meta_data['trace'].append(int(trace.split('_')[-1]))
                        meta_data['sequence'].append(int(seq.split('_')[-1]))
                        meta_data['frame'].append(int(img.split('.')[0]))
                        meta_data['image'].append(img_path)
                        meta_data['control'].append(ctrl_path)
                        meta_data['obstacle'].append(obs_path)
                        meta_data['global_sequence_id'].append(global_seq_id)
                    global_seq_id += 1
        meta_data = pd.DataFrame(data=meta_data)

        train_split_path = os.path.join(data_src, 'train_split.csv')
        if not os.path.exists(train_split_path) or self._args.regenerate_split:
            not_seq_99 = meta_data['sequence'] != 99
            train_split = meta_data.loc[not_seq_99]
            train_split.to_csv(train_split_path)

        val_split_path = os.path.join(data_src, 'val_split.csv')
        if not os.path.exists(val_split_path) or self._args.regenerate_split:
            is_seq_99 = meta_data['sequence'] == 99
            is_seq_38 = meta_data['sequence'] == 38
            val_split = meta_data.loc[is_seq_99 | is_seq_38]
            val_split.to_csv(val_split_path)

        test_split_path = os.path.join(data_src, 'test_split.csv')
        if not os.path.exists(test_split_path) or self._args.regenerate_split:
            is_seq_99 = meta_data['sequence'] == 99
            is_seq_38 = meta_data['sequence'] == 38
            test_split = meta_data.loc[is_seq_99 | is_seq_38]
            test_split.to_csv(test_split_path)


    def setup(self, stage: Optional[str] = None):
        data_src = self._args.data_src

        if stage == "fit" or stage is None:
            train_split_path = os.path.join(data_src, 'train_split.csv')
            train_split = pd.read_csv(train_split_path)
            train_valid_idcs = self._get_valid_idcs(train_split)
            self._train_dset = TorchDataset(train_split, train_valid_idcs, True, self._args)

            val_split_path = os.path.join(data_src, 'val_split.csv')
            val_split = pd.read_csv(val_split_path)
            val_valid_idcs = self._get_valid_idcs(val_split)
            self._val_dset = TorchDataset(val_split, val_valid_idcs, False, self._args)

        if stage == "test" or stage is None:
            test_split_path = os.path.join(data_src, 'test_split.csv')
            test_split = pd.read_csv(test_split_path)
            test_valid_idcs = self._get_valid_idcs(test_split)
            self._test_dset = TorchDataset(test_split, test_valid_idcs, False, self._args)

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

    def test_dataloader(self):
        return DataLoader(self._test_dset,
                          batch_size=self._args.batch_size,
                          num_workers=self._args.num_workers,
                          shuffle=False,
                          pin_memory=True)

    def _get_valid_idcs(self, split):
        if self._args.use_old_indexing:
            valid_idcs = list(range(len(split)))
        else:
            valid_idcs = []
            seq_size = self._args.sequence_size
            n_seqs = split['global_sequence_id'].iloc[-1] + 1
            for seq_id in range(n_seqs):
                mask = split['global_sequence_id'] == seq_id
                seq_data = split.loc[mask]
                filtered_seq_data = seq_data.iloc[seq_size-1:]
                valid_idcs.extend(filtered_seq_data.index.to_list())

        if self._args.exclude_lf_data:
            not_lf_data = set(split.loc[split['trace'] != -1].index)
            valid_idcs = list(set(valid_idcs).intersection(not_lf_data))

        return valid_idcs


class TorchDataset(Dataset):
    def __init__(self, split, valid_idcs, train, args):
        ctrl_paths = split['control'].unique()
        self._control = {k: torch.from_numpy(loadmat(k)['data']) for k in ctrl_paths}

        obs_paths = split['obstacle'].unique()
        self._obstacle = dict()
        for obs_path in obs_paths:
            if obs_path == 'None':
                self._obstacle[obs_path] = None
            else:
                self._obstacle[obs_path] = torch.from_numpy(loadmat(obs_path)['obs_active'])

        self._split = split
        self._valid_idcs = valid_idcs
        self._train = train
        self._args = args

    def __getitem__(self, idx):
        idx = self._valid_idcs[idx]
        if self._args.use_old_indexing:
            idx += self._args.sequence_size - 1 - self._split['frame'][idx]

        metadata = self._split.iloc[idx + 1 - self._args.sequence_size:idx + 1]
        imgs = []
        gamma, brightness, contrast, saturation = None, None, None, None
        for _, row in metadata.iterrows():
            img = read_image(row['image']) / 255.
            img, [gamma, brightness, contrast, saturation] = transform_rgb(img, self._train,
                use_color_jitter=self._args.use_color_jitter,
                use_fixed_standardize=self._args.use_fixed_standardize,
                gamma=gamma, brightness=brightness,
                contrast=contrast, saturation=saturation)
            imgs.append(img)
        imgs = torch.stack(imgs)

        assert len(metadata['control'].unique()) == 1 # DEBUG: sanity check
        frame_start = self._split['frame'][idx] + 1 - self._args.sequence_size
        frame_end = self._split['frame'][idx] + 1
        ctrl_data = self._control[metadata['control'].iloc[-1]][frame_start:frame_end]
        state_data = ctrl_data[:, :6] # s, d, mu, v, delta, kappa
        obs_data = self._obstacle[metadata['obstacle'].iloc[-1]]
        if obs_data is None:
            obs_data = torch.cat([state_data[:, 0].reshape(-1, 1) + self._args.non_obstacle_ds,
                                  torch.ones_like(state_data[:, 1].reshape(-1, 1)) * 8], dim = 1)
        else:
            obs_data = obs_data.expand(state_data.size(0), 2)

        ctrl_label = torch.cat([ctrl_data[:, 3:5], ctrl_data[:, 6:8]], dim=1) # v, delta, a, omega

        state_data = state_data.float()
        obs_data = obs_data.float()
        ctrl_label = ctrl_label.float()

        return imgs, state_data, obs_data, ctrl_label

    def __len__(self):
        return len(self._valid_idcs)
