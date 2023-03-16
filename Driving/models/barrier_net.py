import json
import torch.nn as nn
from torch.autograd import Variable
import torch
from .base import LitModel as Base
from .utils import build_cnn, build_mlp, cvx_solver
from qpth.qp import QPFunction, QPSolvers
import numpy as np


class LitModel(Base):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = Base.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group('models.barrier_net')
        parser.add_argument('--cnn-params', type=json.loads,
                            default=[[3, 24, 5, 2, 2],
                                     [24, 36, 5, 2, 2],
                                     [36, 48, 3, 2, 1],
                                     [48, 64, 3, 1, 1],
                                     [64, 64, 3, 1, 1]])
        parser.add_argument('--cnn-dropout', type=float, default=0.3)
        parser.add_argument('--cnn-with-gn', action='store_true', default=False)
        parser.add_argument('--cnn-activation', type=str, default='relu')
        parser.add_argument('--keep-w-feature', action='store_true', default=False)
        parser.add_argument('--lstm-size', type=int, default=64)
        parser.add_argument('--p-mlp-params', type=json.loads,
                            default=[32, 32, 2])
        parser.add_argument('--p-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--p-mlp-activation', type=str, default='relu')
        parser.add_argument('--p-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--p-mlp-with-gn', action='store_true', default=False)
        parser.add_argument('--p-mlp-no-output-norm', action='store_true', default=False)
        parser.add_argument('--p-lower-bound', type=float, default=None)
        parser.add_argument('--q-mlp-params', type=json.loads,
                            default=[32, 32, 2])
        parser.add_argument('--q-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--q-mlp-activation', type=str, default='relu')
        parser.add_argument('--q-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--q-mlp-with-gn', action='store_true', default=False)
        parser.add_argument('--model-type', type=str, default='deri',
                            choices=['deri', 'inte', 'direct'])
        parser.add_argument('--not-use-gt', action='store_true', default=False)
        parser.add_argument('--randomize-q', action='store_true', default=False)
        parser.add_argument('--detach-q', action='store_true', default=False)
        parser.add_argument('--clip-q', action='store_true', default=False)

        parser.add_argument('--use-state-net', action='store_true', default=False)
        parser.add_argument('--use-indep-state-cnn', action='store_true', default=False)
        parser.add_argument('--state-tol', type=float, default=0.05)
        parser.add_argument('--use-lstm-in-state-net', action='store_true', default=False)
        parser.add_argument('--ds-mlp-params', type=json.loads, default=None)
        parser.add_argument('--ds-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--ds-mlp-activation', type=str, default='relu')
        parser.add_argument('--ds-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--dd-mlp-params', type=json.loads, default=None)
        parser.add_argument('--dd-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--dd-mlp-activation', type=str, default='relu')
        parser.add_argument('--dd-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--d-mlp-params', type=json.loads, default=None)
        parser.add_argument('--d-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--d-mlp-activation', type=str, default='relu')
        parser.add_argument('--d-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--mu-mlp-params', type=json.loads, default=None)
        parser.add_argument('--mu-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--mu-mlp-activation', type=str, default='relu')
        parser.add_argument('--mu-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--kappa-mlp-params', type=json.loads, default=None)
        parser.add_argument('--kappa-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--kappa-mlp-activation', type=str, default='relu')
        parser.add_argument('--kappa-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--obs-d-mlp-params', type=json.loads, default=None)
        parser.add_argument('--obs-d-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--obs-d-mlp-activation', type=str, default='relu')
        parser.add_argument('--obs-d-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--use-lane-keeping-CBFs', action='store_true', default=False)

        return parent_parser

    def custom_setup(self):
        if self.hparams.model_type == 'deri':
            assert self.hparams.output_mode[:4] == ['v', 'delta', 'a', 'omega'], \
                'Type deri of barrier net uses (v, delta, a, omega) as output'
        elif self.hparams.model_type in ['inte', 'direct']:
            assert self.hparams.output_mode[:2] == ['a', 'omega'], \
                'Type inte/direct of barrier net uses (a, omega) as output'
        self._validify_loss_ceof()

        self._cnn = build_cnn(filters=self.hparams.cnn_params,
                              dropout=self.hparams.cnn_dropout,
                              activation=self.hparams.get('cnn_activation', 'relu'),
                              with_gn=self.hparams.get('cnn_with_gn', False))

        if self.hparams.get('keep_w_feature', False):
            cnn_feat_size = self.hparams.cnn_params[-1][0] * 20 # NOTE: hacky way to get feature w
        else:
            cnn_feat_size = self.hparams.cnn_params[-1][0]
        self._lstm = nn.LSTM(cnn_feat_size, self.hparams.lstm_size, batch_first=True)

        p_mlp_params = [self.hparams.lstm_size] + self.hparams.p_mlp_params
        self._p_mlp = build_mlp(filters=p_mlp_params,
                                dropout=self.hparams.p_mlp_dropout,
                                with_bn=self.hparams.p_mlp_with_bn,
                                with_gn=self.hparams.get('p_mlp_with_gn', False),
                                with_output_norm=not self.hparams.get('p_mlp_no_output_norm', False),
                                no_act_last_layer=True,
                                activation=self.hparams.p_mlp_activation)

        q_mlp_params = [self.hparams.lstm_size] + self.hparams.q_mlp_params
        self._q_mlp = build_mlp(filters=q_mlp_params,
                                dropout=self.hparams.q_mlp_dropout,
                                with_bn=self.hparams.q_mlp_with_bn,
                                with_gn=self.hparams.get('p_mlp_with_gn', False),
                                no_act_last_layer=True,
                                activation=self.hparams.q_mlp_activation)

        if self.hparams.get('use_state_net', False):
            assert self.hparams.output_mode[-3:] == ['ds', 'dd', 'mu'] or \
                   self.hparams.output_mode[-4:] == ['ds', 'obs_d', 'd', 'mu'] or \
                   self.hparams.output_mode[-5:] == ['ds', 'obs_d', 'd', 'mu', 'kappa']

            self.hparams.ds_bound = [float(v) for v in self.hparams.ds_bound]
            if self.hparams.ds_bound_for_obs_d:
                self.hparams.ds_bound_for_obs_d = [float(v) for v in self.hparams.ds_bound_for_obs_d]

            if self.hparams.use_indep_state_cnn:
                self._state_cnn = build_cnn(filters=self.hparams.cnn_params,
                                            dropout=self.hparams.cnn_dropout)

                if self.hparams.keep_w_feature:
                    cnn_feat_size = self.hparams.cnn_params[-1][0] * 20 # NOTE: hacky way to get feature w
                else:
                    cnn_feat_size = self.hparams.cnn_params[-1][0]
                self._state_lstm = nn.LSTM(cnn_feat_size,
                                        self.hparams.lstm_size,
                                        batch_first=True)

            feat_size = self.hparams.lstm_size if \
                self.hparams.use_lstm_in_state_net else cnn_feat_size

            if self.hparams.ds_mlp_params:
                self._ds_mlp = build_mlp(filters=[feat_size] + self.hparams.ds_mlp_params,
                                         dropout=self.hparams.ds_mlp_dropout,
                                         with_bn=self.hparams.ds_mlp_with_bn,
                                         no_act_last_layer=True,
                                         activation=self.hparams.ds_mlp_activation)

            if self.hparams.dd_mlp_params:
                self._dd_mlp = build_mlp(filters=[feat_size] + self.hparams.dd_mlp_params,
                                         dropout=self.hparams.dd_mlp_dropout,
                                         with_bn=self.hparams.dd_mlp_with_bn,
                                         no_act_last_layer=True,
                                         activation=self.hparams.dd_mlp_activation)

            if self.hparams.d_mlp_params:
                self._d_mlp = build_mlp(filters=[feat_size] + self.hparams.d_mlp_params,
                                        dropout=self.hparams.d_mlp_dropout,
                                        with_bn=self.hparams.d_mlp_with_bn,
                                        no_act_last_layer=True,
                                        activation=self.hparams.d_mlp_activation)

            if self.hparams.mu_mlp_params:
                self._mu_mlp = build_mlp(filters=[feat_size] + self.hparams.mu_mlp_params,
                                         dropout=self.hparams.mu_mlp_dropout,
                                         with_bn=self.hparams.mu_mlp_with_bn,
                                         no_act_last_layer=True,
                                         activation=self.hparams.mu_mlp_activation)

            if self.hparams.kappa_mlp_params:
                self._kappa_mlp = build_mlp(filters=[feat_size] + self.hparams.kappa_mlp_params,
                                            dropout=self.hparams.kappa_mlp_dropout,
                                            with_bn=self.hparams.kappa_mlp_with_bn,
                                            no_act_last_layer=True,
                                            activation=self.hparams.kappa_mlp_activation)

            if self.hparams.obs_d_mlp_params:
                self._obs_d_mlp = build_mlp(filters=[feat_size] + self.hparams.obs_d_mlp_params,
                                            dropout=self.hparams.obs_d_mlp_dropout,
                                            with_bn=self.hparams.obs_d_mlp_with_bn,
                                            no_act_last_layer=True,
                                            activation=self.hparams.obs_d_mlp_activation)

    def forward(self, batch, rnn_state, solver='QP', store_intermediate_data=False, **kwargs):
        img_seq = batch[0]
        state_seq = batch[1]
        obs_seq = batch[2]
        ctrl_seq = batch[3]

        b, t, c, h, w = img_seq.shape
        img_seq_fl = img_seq.view(b * t, c, h, w)
        z = self._cnn(img_seq_fl)
        if self.hparams.get('keep_w_feature', False):
            z = z.max(-2)[0] # max-pooling over h dim only
            z = z.flatten(-2, -1) # flatten c and h
        else:
            z = z.max(-1)[0].max(-1)[0] # max-pooling over h and w dims
        cnn_feat = z
        z = z.view(b, t, -1)

        z, rnn_state = self._lstm(z, rnn_state)

        z = z.reshape(b * t, -1)
        q = self._q_mlp(z)
        p = self._p_mlp(z)*4

        if self.hparams.p_lower_bound is not None:
            p = torch.clamp(p, min=self.hparams.p_lower_bound)

        if solver == 'cvxpy':
            q = q.view(b, t, -1)
            q = q[:,-1]
            p = p.view(b, t, -1)
            p = p[:,-1]
            state_seq = state_seq[:,-1]
            obs_seq = obs_seq[:,-1]
            ctrl_seq = ctrl_seq[:,-1]
            nBatch = b
        else:
            nBatch = b*t
            state_seq = state_seq.view(nBatch, -1)
            obs_seq = obs_seq.view(nBatch, -1)
            ctrl_seq = ctrl_seq.view(nBatch, -1)

        #softened HOCBFs for safety
        if self.hparams.model_type in ['deri', 'deri_ref']:
            v_delta = q
            if solver == 'cvxpy' or self.hparams.not_use_gt:
                q = -((q - state_seq[:,3:5])/0.1)

                if self.hparams.detach_q:
                    q = q.detach()

                if self.hparams.clip_q or solver == 'cvxpy': # NOTE: bound q during eval
                    q[:, 0] = torch.clamp(q[:, 0], -1.0, 1.0) # -omega
                    q[:, 1] = torch.clamp(q[:, 1], -7.0, 7.0) # -a
            elif self.hparams.randomize_q and self.training:
                q = torch.stack([(torch.rand(q[:, 0].shape) - 0.5) * 1,
                                 (torch.rand(q[:, 1].shape) - 0.5) * 7], dim=1).to(q)
            else:
                q = -ctrl_seq[:,2:4]
        elif self.hparams.model_type in ['inte', 'direct']:
            pass # reference control q is a-omega

        # Set up the cost of the neuron of BarrierNet
        q_size = self.hparams.q_mlp_params[-1]
        Q = Variable(torch.eye(q_size))
        Q = Q.unsqueeze(0).expand(nBatch, q_size, q_size).to(self.device)

        gt_s, gt_d, gt_mu, v, delta, gt_kappa = state_seq[:,0], state_seq[:,1], \
                state_seq[:,2], state_seq[:,3], state_seq[:,4], state_seq[:,5]
        gt_obs_s, gt_obs_d = obs_seq[:,0], obs_seq[:,1]
        gt_ds = gt_s - gt_obs_s
        gt_dd = gt_d - gt_obs_d
        if self.hparams.get('use_state_net', False):
            if self.hparams.use_indep_state_cnn:
                state_z = self._state_cnn(img_seq_fl)
                if self.hparams.keep_w_feature:
                    state_z = state_z.max(-2)[0]  # max-pooling over h dim only
                    state_z = state_z.flatten(-2, -1)  # flatten c and h
                else:
                    state_z = state_z.max(-1)[0].max(-1)[0]  # max-pooling over h and w dims
                cnn_feat = state_z
                state_z = state_z.view(b, t, -1)
                state_z, rnn_state = self._lstm(state_z, rnn_state)
            else:
                state_z = z

            state_net_in = state_z if self.hparams.use_lstm_in_state_net else cnn_feat

            d = self._d_mlp(state_net_in).squeeze(1) if self.hparams.d_mlp_params else gt_d
            mu = self._mu_mlp(state_net_in).squeeze(1) if self.hparams.mu_mlp_params else gt_mu
            kappa = self._kappa_mlp(state_net_in).squeeze(1) if self.hparams.kappa_mlp_params else gt_kappa
            obs_d = self._obs_d_mlp(state_net_in).squeeze(1) if self.hparams.d_mlp_params else gt_obs_d
            if self.hparams.drop_obs_d_offset:
                obs_d = obs_d + torch.sign(obs_d) * 5
            ds = self._ds_mlp(state_net_in).squeeze(1) if self.hparams.ds_mlp_params else gt_s - gt_obs_s
            dd = self._dd_mlp(state_net_in).squeeze(1) if self.hparams.dd_mlp_params else d - obs_d
            if self.hparams.drop_obs_d_offset:
                dd = dd + torch.sign(dd) * 5

            d_non_detach = d
            mu_non_detach = mu
            kappa_non_detach = kappa
            obs_d_non_detach = obs_d
            ds_non_detach = ds
            dd_non_detach = dd

            tol = self.hparams.state_tol
            d = torch.clamp(d.detach(), gt_d * (1 - tol), gt_d * (1 + tol))
            mu = torch.clamp(mu.detach(), gt_mu * (1 - tol), gt_mu * (1 + tol))
            kappa = torch.clamp(kappa.detach(), gt_kappa * (1 - tol), gt_kappa * (1 + tol))
            obs_d = torch.clamp(obs_d.detach(), gt_obs_d * (1 - tol), gt_obs_d * (1 + tol))
            ds = torch.clamp(ds.detach(), gt_ds * (1 - tol), gt_ds * (1 + tol))
            dd = torch.clamp(dd.detach(), gt_dd * (1 - tol), gt_dd * (1 + tol))
        elif self.hparams.get('use_indep_state_net', False):
            indep_state_pred = batch[4].reshape(b * t, -1)
            if self.hparams.indep_state_net_output == ['ds', 'd', 'mu', 'kappa', 'obs_d']:
                ds = indep_state_pred[:,0]
                d = indep_state_pred[:,1]
                mu = indep_state_pred[:,2]
                kappa = indep_state_pred[:,3]
                obs_d = indep_state_pred[:,4]
                dd = d - obs_d
            elif self.hparams.indep_state_net_output == ['ds', 'obs_d', 'd', 'mu']:
                ds = indep_state_pred[:,0]
                obs_d = indep_state_pred[:,1]
                d = indep_state_pred[:,2]
                mu = indep_state_pred[:,3]
                kappa =  torch.zeros_like(gt_kappa) # NOTE: set to 0
                dd = d - obs_d
            elif self.hparams.indep_state_net_output == ['ds', 'dd', 'mu']:
                ds = indep_state_pred[:,0]
                dd = indep_state_pred[:,1]
                mu = indep_state_pred[:,2]
                kappa = torch.zeros_like(gt_kappa) # NOTE: set to 0
                d = gt_d # NOTE: doens't matter as kappa = 0
            else:
                raise NotImplementedError(f'Unrecognized state net output {self.hparams.indep_state_net_output}')
        else:
            s, d, mu, v, delta, kappa = gt_s, gt_d, gt_mu, v, delta, gt_kappa
            obs_s, obs_d = gt_obs_s, gt_obs_d
            ds, dd = gt_ds, gt_dd

        lrf, lr = 0.5, 2.78 #lr/(lr+lf)
        beta = torch.atan(lrf*torch.tan(delta))
        cos_mu_beta = torch.cos(mu + beta)
        sin_mu_beta = torch.sin(mu + beta)
        mu_dot = v/lr*torch.sin(beta) - kappa*v*cos_mu_beta/(1 - d*kappa)

        barrier = ds**2 + dd**2 - 7.9**2  #radius of the obstacle-covering disk is 7.9 < 8m (mpc), avoiding the set boundary
        barrier_dot = 2*ds*v*cos_mu_beta/(1 - d*kappa) + 2*dd*v*sin_mu_beta
        Lf2b = 2*(v*cos_mu_beta/(1 - d*kappa))**2 + 2*(v*sin_mu_beta)**2 - 2*ds*v*sin_mu_beta*mu_dot/(1 - d*kappa)  + 2*ds*kappa*v**2*sin_mu_beta*cos_mu_beta/(1 - d*kappa)**2 + 2*dd*v*cos_mu_beta*mu_dot
        LgLfbu1 = 2*ds*cos_mu_beta/(1 - d*kappa) + 2*dd*sin_mu_beta
        LgLfbu2 = (-2*ds*v*sin_mu_beta/(1 - d*kappa) + 2*dd*v*cos_mu_beta)*lrf/torch.cos(delta)**2/(1 + (lrf*torch.tan(delta))**2)
        LgLfbu1 = torch.reshape(LgLfbu1, (nBatch, 1))
        LgLfbu2 = torch.reshape(LgLfbu2, (nBatch, 1))

        G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        G = torch.reshape(G, (nBatch, 1, 2)).to(self.device)
        h = (torch.reshape((Lf2b + (p[:,0] + p[:,1])*barrier_dot + p[:,0]*p[:,1]*barrier), (nBatch, 1))).to(self.device)
        e = Variable(torch.Tensor()).to(self.device) #no equality constraints
        
        if self.hparams.get('use_lane_keeping_CBFs', True):
            # softened HOCBFs for lane keeping - left
            barrier = self.hparams.get('lf_cbf_threshold', 2.0) - d
            barrier_dot = -v*sin_mu_beta
            Lf2b = -v*cos_mu_beta*mu_dot
            LgLfbu1 = -sin_mu_beta
            LgLfbu2 = -v*cos_mu_beta*lrf/torch.cos(delta)**2/(1 + (lrf*torch.tan(delta))**2)
            LgLfbu1 = torch.reshape(LgLfbu1, (nBatch, 1))
            LgLfbu2 = torch.reshape(LgLfbu2, (nBatch, 1))
            G1 = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
            G1 = torch.reshape(G1, (nBatch, 1, 2))
            h1 = (torch.reshape((Lf2b + 2*barrier_dot + 1*barrier), (nBatch, 1)))

            # softened HOCBFs for lane keeping - right
            barrier = d + self.hparams.get('lf_cbf_threshold', 2.0)
            barrier_dot = v*sin_mu_beta
            Lf2b = v*cos_mu_beta*mu_dot
            LgLfbu1 = sin_mu_beta
            LgLfbu2 = v*cos_mu_beta*lrf/torch.cos(delta)**2/(1 + (lrf*torch.tan(delta))**2)
            LgLfbu1 = torch.reshape(LgLfbu1, (nBatch, 1))
            LgLfbu2 = torch.reshape(LgLfbu2, (nBatch, 1))
            G2 = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
            G2 = torch.reshape(G2, (nBatch, 1, 2))
            h2 = (torch.reshape((Lf2b + 2*barrier_dot + 1*barrier), (nBatch, 1)))
            if self.hparams.use_lf_cbf_only:
                G = torch.cat([G1,G2], dim = 1).to(self.device)
                h = torch.cat([h1,h2], dim = 1).to(self.device)
            else:
                G = torch.cat([G,G1,G2], dim = 1).to(self.device)
                h = torch.cat([h,h1,h2], dim = 1).to(self.device)
        if solver == 'cvxpy':
            self.p1 = p[0,0]
            self.p2 = p[0,1]
            x = cvx_solver(Q[0].double(), q[0].double(), G[0].double(), h[0].double())
            x = np.array([[x[0], x[1]]])
            x = torch.tensor(x).float().to(self.device)
        else:
            x = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        if self.hparams.model_type == 'deri':
            out = torch.cat([v_delta, x], dim = 1)
        elif self.hparams.model_type == 'deri_ref':
            out = torch.cat([v_delta, -q], dim = 1)
        elif self.hparams.model_type == 'inte':
            out = state_seq[:,3:5] + x*0.1
        elif self.hparams.model_type == 'direct':
            out = x

        if self.hparams.get('use_state_net', False):
            if self.hparams.output_mode[-3:] == ['ds', 'dd', 'mu']:
                state_out = torch.stack([ds_non_detach, dd_non_detach, mu_non_detach], dim=1)
            elif self.hparams.output_mode[-4:] == ['ds', 'obs_d', 'd', 'mu']:
                state_out = torch.stack([ds_non_detach, obs_d_non_detach, d_non_detach, mu_non_detach], dim=1)
            elif self.hparams.output_mode[-5:] == ['ds', 'obs_d', 'd', 'mu', 'kappa']:
                state_out = torch.stack([ds_non_detach, obs_d_non_detach, d_non_detach, mu_non_detach, kappa_non_detach], dim=1)

            out = torch.cat([out, state_out], dim=1)

        out = out.view(b, t, -1)

        if store_intermediate_data: # NOTE: store as scalars
            self.intermediate_data = {
                'q0': q[0, 0].item(),
                'q1': q[0, 1].item(),
                'p0': p[0, 0].item(),
                'p1': p[0, 1].item(),
                'gt_s': gt_s.item(),
                'gt_d': gt_d.item(),
                'gt_mu': gt_mu.item(),
                'gt_v': v.item(),
                'gt_delta': delta.item(),
                'gt_kappa': gt_kappa.item(),
                'gt_obs_s': gt_obs_s[0].item(),
                'gt_obs_d': gt_obs_d[0].item(),
                'gt_ds': gt_ds[0].item(),
                'gt_dd': gt_dd[0].item(),
                'barrier': barrier.item(),
                'Gz': (G * x).sum().item(),
                'h': h[0, 0].item(),
                'active': not torch.allclose(x, -q),
            }
            if 'ds' in locals().keys():
                self.intermediate_data['ds'] = ds[0].item()
            if 'dd' in locals().keys():
                self.intermediate_data['dd'] = dd[0].item()
            if 'ds' in self.hparams.output_mode or 'ds' in self.hparams.get('indep_state_net_output', []):
                self.intermediate_data['pred_ds'] = ds[0].item()
            if 'dd' in self.hparams.output_mode or 'dd' in self.hparams.get('indep_state_net_output', []):
                self.intermediate_data['pred_dd'] = dd[0].item()
            if 'd' in self.hparams.output_mode or 'd' in self.hparams.get('indep_state_net_output', []):
                self.intermediate_data['pred_d'] = d[0].item()
            if 'mu' in self.hparams.output_mode or 'mu' in self.hparams.get('indep_state_net_output', []):
                self.intermediate_data['pred_mu'] = mu[0].item()
            if 'kappa' in self.hparams.output_mode or 'kappa' in self.hparams.get('indep_state_net_output', []):
                self.intermediate_data['pred_kappa'] = kappa[0].item()
            if 'obs_d' in self.hparams.output_mode or 'obs_d' in self.hparams.get('indep_state_net_output', []):
                self.intermediate_data['pred_obs_d'] = obs_d[0].item()

        return out, rnn_state

    def get_initial_state(self, batch_size):
        state_size = (1, batch_size, self._lstm.hidden_size) # num_layer x bsize x hidden size
        h_state = torch.zeros(state_size, dtype=torch.float32).to(self.device)
        c_state = torch.zeros(state_size, dtype=torch.float32).to(self.device)
        return [h_state, c_state]
