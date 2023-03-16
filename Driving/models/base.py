from typing import Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl


class LitModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters() # store kwargs to self.hparams
        self.custom_setup()

        self._loss_fn = nn.MSELoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('models.base')
        parser.add_argument('--every-n-epochs', type=int, default=1)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--output-mode', nargs='+', default=['v', 'delta'],
                            choices=['v', 'delta', 'a', 'omega',
                                     'ds', 'dd', 'd', 'mu', 'kappa', 'obs_d'])
        # control
        parser.add_argument('--loss-coef-v', type=float, default=None)
        parser.add_argument('--loss-coef-delta', type=float, default=None)
        parser.add_argument('--loss-coef-a', type=float, default=None)
        parser.add_argument('--loss-coef-omega', type=float, default=None)
        # state
        parser.add_argument('--loss-coef-ds', type=float, default=None)
        parser.add_argument('--loss-coef-dd', type=float, default=None)
        parser.add_argument('--loss-coef-d', type=float, default=None)
        parser.add_argument('--loss-coef-mu', type=float, default=None)
        parser.add_argument('--loss-coef-kappa', type=float, default=None)
        parser.add_argument('--loss-coef-obs-d', type=float, default=None)
        parser.add_argument('--ds-bound', nargs='+', default=[-5., 40.])
        parser.add_argument('--ds-bound-for-obs-d', nargs='+', default=None)
        parser.add_argument('--drop-obs-d-offset', action='store_true', default=False)

        return parent_parser

    def custom_setup(self):
        raise NotImplementedError('To be defined in child class')

    def forward(self, batch, rnn_state, **kwargs):
        raise NotImplementedError('To be defined in child class')

    def training_step(self, batch, batch_idx):
        if hasattr(self, 'get_initial_state'):
            batch_size = batch[0].shape[0]
            rnn_state = self.get_initial_state(batch_size)
        else:
            rnn_state = None

        pred, rnn_state = self(batch, rnn_state, solver = 'QP')
        losses = self._compute_loss(pred, batch)

        for k, v in losses.items():
            self.log(f'train/{k}', v)

        return losses['loss/all']

    def validation_step(self, batch, batch_idx):
        if hasattr(self, 'get_initial_state'):
            batch_size = batch[0].shape[0]
            rnn_state = self.get_initial_state(batch_size)
        else:
            rnn_state = None

        pred, rnn_state = self(batch, rnn_state, solver = 'QP')
        losses = self._compute_loss(pred, batch)

        for k, v in losses.items():
            self.log(f'val/{k}', v)

    def test_step(self, batch, batch_idx):
        # TODO: unused for now; different logging from validation_step
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _validify_loss_ceof(self):
        active_loss = []
        for k, v in self.hparams.items():
            if 'loss_coef' in k and v is not None:
                active_loss.append(k.split('loss_coef_')[-1])
        output_mode = self.hparams.output_mode
        assert set(active_loss) == set(output_mode), \
            f'Active loss coefficients {active_loss} not consistent with output mode {output_mode}'

    def _compute_loss(self, pred, batch):
        state = batch[1]
        obs = batch[2]
        label = batch[3]

        losses = dict()
        loss_all = 0.

        if 'v' in self.hparams.output_mode:
            pred_v = pred[:, :, self.hparams.output_mode.index('v')]
            label_v = label[:, :, 0]
            loss_v = self._loss_fn(pred_v, label_v)
            loss_all += self.hparams.loss_coef_v * loss_v
            losses['loss/v'] = loss_v

        if 'delta' in self.hparams.output_mode:
            pred_delta = pred[:, :, self.hparams.output_mode.index('delta')]
            label_delta = label[:, :, 1]
            loss_delta = self._loss_fn(pred_delta, label_delta)
            loss_all += self.hparams.loss_coef_delta * loss_delta
            losses['loss/delta'] = loss_delta

        if 'a' in self.hparams.output_mode:
            pred_a = pred[:, :, self.hparams.output_mode.index('a')]
            label_a = label[:, :, 2]
            loss_a = self._loss_fn(pred_a, label_a)
            loss_all += self.hparams.loss_coef_a * loss_a
            losses['loss/a'] = loss_a

        if 'omega' in self.hparams.output_mode:
            pred_omega = pred[:, :, self.hparams.output_mode.index('omega')]
            label_omega = label[:, :, 3]
            loss_omega = self._loss_fn(pred_omega, label_omega)
            loss_all += self.hparams.loss_coef_omega * loss_omega
            losses['loss/omega'] = loss_omega

        if 'ds' in self.hparams.output_mode:
            pred_ds = pred[:, :, self.hparams.output_mode.index('ds')]
            label_ds = state[:, :, 0] - obs[:, :, 0]

            ds_larger = pred_ds > self.hparams.ds_bound[1]
            upper_mask = torch.logical_and(label_ds > self.hparams.ds_bound[1], ds_larger)
            ds_smaller = pred_ds < self.hparams.ds_bound[0]
            lower_mask = torch.logical_and(label_ds < self.hparams.ds_bound[0], ds_smaller)
            mask = torch.logical_not(torch.logical_or(upper_mask, lower_mask))

            original_reduction = self._loss_fn.reduction
            self._loss_fn.reduction = 'none'
            loss_ds = self._loss_fn(pred_ds, label_ds)
            self._loss_fn.reduction = original_reduction
            loss_ds = (loss_ds * mask.float()).mean()

            loss_all += self.hparams.loss_coef_ds * loss_ds
            losses['loss/ds'] = loss_ds

        if 'dd' in self.hparams.output_mode:
            pred_dd = pred[:, :, self.hparams.output_mode.index('dd')]
            if self.hparams.drop_obs_d_offset:
                obs_d_no_offset = obs[:, :, 1] - torch.sign(obs[:, :, 1]) * 5
                label_dd = state[:, :, 1] - obs_d_no_offset
            else:
                label_dd = state[:, :, 1] - obs[:, :, 1]

            if self.hparams.ds_bound_for_obs_d:
                label_ds = state[:, :, 0] - obs[:, :, 0]
                mask = torch.logical_and(label_ds > self.hparams.ds_bound_for_obs_d[0],
                                         label_ds < self.hparams.ds_bound_for_obs_d[1])

                original_reduction = self._loss_fn.reduction
                self._loss_fn.reduction = 'none'
                loss_dd = self._loss_fn(pred_dd, label_dd)
                self._loss_fn.reduction = original_reduction
                loss_dd = (loss_dd * mask.float()).mean()
            else:
                loss_dd = self._loss_fn(pred_dd, label_dd)
            loss_all += self.hparams.loss_coef_ds * loss_dd
            losses['loss/dd'] = loss_dd

        if 'd' in self.hparams.output_mode:
            pred_d = pred[:, :, self.hparams.output_mode.index('d')]
            label_d = state[:, :, 1]
            loss_d = self._loss_fn(pred_d, label_d)
            loss_all += self.hparams.loss_coef_d * loss_d
            losses['loss/d'] = loss_d

        if 'mu' in self.hparams.output_mode:
            pred_mu = pred[:, :, self.hparams.output_mode.index('mu')]
            label_mu = state[:, :, 2]
            loss_mu = self._loss_fn(pred_mu, label_mu)
            loss_all += self.hparams.loss_coef_mu * loss_mu
            losses['loss/mu'] = loss_mu

        if 'kappa' in self.hparams.output_mode:
            pred_kappa = pred[:, :, self.hparams.output_mode.index('kappa')]
            label_kappa = state[:, :, 5]
            loss_kappa = self._loss_fn(pred_kappa, label_kappa)
            loss_all += self.hparams.loss_coef_kappa * loss_kappa
            losses['loss/kappa'] = loss_kappa

        if 'obs_d' in self.hparams.output_mode:
            pred_obs_d = pred[:, :, self.hparams.output_mode.index('obs_d')]
            if self.hparams.drop_obs_d_offset:
                obs_d_no_offset = obs[:, :, 1] - torch.sign(obs[:, :, 1]) * 5
                label_obs_d = obs_d_no_offset
            else:
                label_obs_d = obs[:, :, 1]

            if self.hparams.ds_bound_for_obs_d:
                label_ds = state[:, :, 0] - obs[:, :, 0]
                mask = torch.logical_and(label_ds > self.hparams.ds_bound_for_obs_d[0],
                                         label_ds < self.hparams.ds_bound_for_obs_d[1])

                original_reduction = self._loss_fn.reduction
                self._loss_fn.reduction = 'none'
                loss_obs_d = self._loss_fn(pred_obs_d, label_obs_d)
                self._loss_fn.reduction = original_reduction
                loss_obs_d = (loss_obs_d * mask.float()).mean()
            else:
                loss_obs_d = self._loss_fn(pred_obs_d, label_obs_d)
            loss_all += self.hparams.loss_coef_obs_d * loss_obs_d
            losses['loss/obs_d'] = loss_obs_d

        losses['loss/all'] = loss_all

        return losses
