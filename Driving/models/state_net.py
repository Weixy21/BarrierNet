import json
import torch
import torch.nn as nn
from .base import LitModel as Base
from .utils import build_cnn, build_mlp


class LitModel(Base):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = Base.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group('models.state_net')
        parser.add_argument('--cnn-params',
                            type=json.loads,
                            default=[[3, 24, 5, 2, 2], [24, 36, 5, 2, 2],
                                     [36, 48, 3, 2, 1], [48, 64, 3, 1, 1],
                                     [64, 64, 3, 1, 1]])
        parser.add_argument('--cnn-dropout', type=float, default=0.3)
        parser.add_argument('--cnn-with-gn', action='store_true', default=False)
        parser.add_argument('--cnn-activation', type=str, default='relu')
        parser.add_argument('--keep-w-feature', action='store_true', default=False)
        parser.add_argument('--use-lstm', action='store_true', default=False)
        parser.add_argument('--use-indep-lstm', action='store_true', default=False)
        parser.add_argument('--lstm-size', type=int, default=64)

        parser.add_argument('--mlp-params', type=json.loads, default=[64, 64, 5])
        parser.add_argument('--mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--mlp-activation', type=str, default='relu')
        parser.add_argument('--mlp-dropout', type=float, default=0.3)

        parser.add_argument('--use-indep-mlps', action='store_true', default=False)
        parser.add_argument('--ds-mlp-params', type=json.loads, default=[32, 32, 1])
        parser.add_argument('--ds-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--ds-mlp-with-gn', action='store_true', default=False)
        parser.add_argument('--ds-mlp-activation', type=str, default='elu')
        parser.add_argument('--ds-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--dd-mlp-params', type=json.loads, default=[32, 32, 1])
        parser.add_argument('--dd-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--dd-mlp-with-gn', action='store_true', default=False)
        parser.add_argument('--dd-mlp-activation', type=str, default='elu')
        parser.add_argument('--dd-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--d-mlp-params', type=json.loads, default=[32, 32, 1])
        parser.add_argument('--d-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--d-mlp-with-gn', action='store_true', default=False)
        parser.add_argument('--d-mlp-activation', type=str, default='elu')
        parser.add_argument('--d-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--mu-mlp-params', type=json.loads, default=[32, 32, 1])
        parser.add_argument('--mu-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--mu-mlp-with-gn', action='store_true', default=False)
        parser.add_argument('--mu-mlp-activation', type=str, default='elu')
        parser.add_argument('--mu-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--kappa-mlp-params', type=json.loads, default=[32, 32, 1])
        parser.add_argument('--kappa-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--kappa-mlp-with-gn', action='store_true', default=False)
        parser.add_argument('--kappa-mlp-activation', type=str, default='elu')
        parser.add_argument('--kappa-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--obs-d-mlp-params', type=json.loads, default=[32, 32, 1])
        parser.add_argument('--obs-d-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--obs-d-mlp-with-gn', action='store_true', default=False)
        parser.add_argument('--obs-d-mlp-activation', type=str, default='elu')
        parser.add_argument('--obs-d-mlp-dropout', type=float, default=0.3)

        parser.add_argument('--output-scale-ds', type=float, default=10.)
        parser.add_argument('--output-scale-dd', type=float, default=1.)
        parser.add_argument('--output-scale-d', type=float, default=1.)
        parser.add_argument('--output-scale-mu', type=float, default=1.)
        parser.add_argument('--output-scale-kappa', type=float, default=1.)
        parser.add_argument('--output-scale-obs-d', type=float, default=1.)

        return parent_parser

    def custom_setup(self):
        assert self.hparams.output_mode == ['ds', 'dd', 'mu'] or \
               self.hparams.output_mode == ['ds', 'obs_d', 'd', 'mu'] or \
               self.hparams.output_mode == ['ds', 'obs_d', 'd', 'mu', 'kappa']
        assert len(self.hparams.ds_bound) == 2
        self._validify_loss_ceof()

        self.hparams.ds_bound = [float(v) for v in self.hparams.ds_bound]
        if self.hparams.ds_bound_for_obs_d:
            self.hparams.ds_bound_for_obs_d = [float(v) for v in self.hparams.ds_bound_for_obs_d]

        self._cnn = build_cnn(filters=self.hparams.cnn_params,
                              dropout=self.hparams.cnn_dropout,
                              activation=self.hparams.get('cnn_activation', 'relu'),
                              with_gn=self.hparams.get('cnn_with_gn', False))

        if self.hparams.keep_w_feature:
            cnn_feat_size = self.hparams.cnn_params[-1][0] * 20 # NOTE: hacky way to get feature w
        else:
            cnn_feat_size = self.hparams.cnn_params[-1][0]

        if self.hparams.use_indep_lstm:
            for out_name in self.hparams.output_mode:
                lstm = nn.LSTM(cnn_feat_size, self.hparams.lstm_size, batch_first=True)
                setattr(self, f'_{out_name}_lstm', lstm)
        elif self.hparams.use_lstm:
            self._lstm = nn.LSTM(cnn_feat_size,
                                 self.hparams.lstm_size,
                                 batch_first=True)

        feat_size = self.hparams.lstm_size if self.hparams.use_lstm or self.hparams.use_indep_lstm else cnn_feat_size
        if self.hparams.get('use_indep_mlps', False):
            if 'ds' in self.hparams.output_mode:
                self._ds_mlp = build_mlp(filters=[feat_size] + self.hparams.ds_mlp_params,
                                         dropout=self.hparams.ds_mlp_dropout,
                                         with_bn=self.hparams.ds_mlp_with_bn,
                                         with_gn=self.hparams.get('ds_mlp_with_gn', False),
                                         no_act_last_layer=True,
                                         activation=self.hparams.ds_mlp_activation)

            if 'dd' in self.hparams.output_mode:
                self._dd_mlp = build_mlp(filters=[feat_size] + self.hparams.dd_mlp_params,
                                         dropout=self.hparams.dd_mlp_dropout,
                                         with_bn=self.hparams.dd_mlp_with_bn,
                                         with_gn=self.hparams.get('dd_mlp_with_gn', False),
                                         no_act_last_layer=True,
                                         activation=self.hparams.dd_mlp_activation)

            if 'd' in self.hparams.output_mode:
                self._d_mlp = build_mlp(filters=[feat_size] + self.hparams.d_mlp_params,
                                        dropout=self.hparams.d_mlp_dropout,
                                        with_bn=self.hparams.d_mlp_with_bn,
                                        with_gn=self.hparams.get('d_mlp_with_gn', False),
                                        no_act_last_layer=True,
                                        activation=self.hparams.d_mlp_activation)

            if 'mu' in self.hparams.output_mode:
                self._mu_mlp = build_mlp(filters=[feat_size] + self.hparams.mu_mlp_params,
                                         dropout=self.hparams.mu_mlp_dropout,
                                         with_bn=self.hparams.mu_mlp_with_bn,
                                         with_gn=self.hparams.get('mu_mlp_with_gn', False),
                                         no_act_last_layer=True,
                                         activation=self.hparams.mu_mlp_activation)

            if 'kappa' in self.hparams.output_mode:
                self._kappa_mlp = build_mlp(filters=[feat_size] + self.hparams.kappa_mlp_params,
                                            dropout=self.hparams.kappa_mlp_dropout,
                                            with_bn=self.hparams.kappa_mlp_with_bn,
                                            with_gn=self.hparams.get('kappa_mlp_with_gn', False),
                                            no_act_last_layer=True,
                                            activation=self.hparams.kappa_mlp_activation)

            if 'obs_d' in self.hparams.output_mode:
                self._obs_d_mlp = build_mlp(filters=[feat_size] + self.hparams.obs_d_mlp_params,
                                            dropout=self.hparams.obs_d_mlp_dropout,
                                            with_bn=self.hparams.obs_d_mlp_with_bn,
                                            with_gn=self.hparams.get('obs_d_mlp_with_gn', False),
                                            no_act_last_layer=True,
                                            activation=self.hparams.obs_d_mlp_activation)
        else:
            mlp_params = [feat_size] + self.hparams.mlp_params

            self._mlp = build_mlp(filters=mlp_params,
                                dropout=self.hparams.mlp_dropout,
                                with_bn=self.hparams.mlp_with_bn,
                                activation=self.hparams.mlp_activation,
                                no_act_last_layer=True)

    def forward(self, batch, rnn_state, **kwargs):
        img_seq = batch[0]
        b, t, c, h, w = img_seq.shape
        img_seq_fl = img_seq.view(b * t, c, h, w)
        z = self._cnn(img_seq_fl)
        if self.hparams.keep_w_feature:
            z = z.max(-2)[0]  # max-pooling over h dim only
            z = z.flatten(-2, -1)  # flatten c and h
        else:
            z = z.max(-1)[0].max(-1)[0]  # max-pooling over h and w dims
        z = z.view(b, t, -1)

        if self.hparams.get('use_indep_lstm', False):
            lstm_out = dict()
            for out_name in self.hparams.output_mode:
                lstm_out[out_name], rnn_state[out_name] = getattr(self, f'_{out_name}_lstm')(z, rnn_state[out_name])
                lstm_out[out_name] = lstm_out[out_name].reshape(b * t, -1)
        else:
            if self.hparams.use_lstm:
                z, rnn_state = self._lstm(z, rnn_state)
            z = z.reshape(b * t, -1)

        if self.hparams.get('use_indep_mlps', False):
            out = []
            for out_name in self.hparams.output_mode:
                if self.hparams.get('use_indep_lstm', False):
                    mlp_in = lstm_out[out_name]
                else:
                    mlp_in = z
                out.append(getattr(self, f'_{out_name}_mlp')(mlp_in))
            out = torch.cat(out, dim=1).view(b, t, -1)
        else:
            z = self._mlp(z)
            out = z.view(b, t, -1)

        if 'ds' in self.hparams.output_mode:
            out[:, :, self.hparams.output_mode.index('ds')] *= self.hparams.output_scale_ds
        if 'dd' in self.hparams.output_mode:
            out[:, :, self.hparams.output_mode.index('dd')] *= self.hparams.output_scale_dd
        if 'd' in self.hparams.output_mode:
            out[:, :, self.hparams.output_mode.index('d')] *= self.hparams.output_scale_d
        if 'mu' in self.hparams.output_mode:
            out[:, :, self.hparams.output_mode.index('mu')] *= self.hparams.output_scale_mu
        if 'kappa' in self.hparams.output_mode:
            out[:, :, self.hparams.output_mode.index('kappa')] *= self.hparams.output_scale_kappa
        if 'obs-d' in self.hparams.output_mode:
            out[:, :, self.hparams.output_mode.index('obs_d')] *= self.hparams.output_scale_obs_d

        return out, rnn_state

    def get_initial_state(self, batch_size):
        if self.hparams.get('use_indep_lstm', False):
            state = dict()
            for out_name in self.hparams.output_mode:
                state_size = (1, batch_size, getattr(self, f'_{out_name}_lstm').hidden_size)
                h_state = torch.zeros(state_size, dtype=torch.float32).to(self.device)
                c_state = torch.zeros(state_size, dtype=torch.float32).to(self.device)
                state[out_name] = [h_state, c_state]
            return state
        elif self.hparams.use_lstm:
            state_size = (1, batch_size, self._lstm.hidden_size
                        )  # num_layer x bsize x hidden size
            h_state = torch.zeros(state_size, dtype=torch.float32).to(self.device)
            c_state = torch.zeros(state_size, dtype=torch.float32).to(self.device)
            return [h_state, c_state]
        else:
            return None
