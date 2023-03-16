import json
import torch
import torch.nn as nn
from .base import LitModel as Base
from .utils import build_cnn, build_mlp


class LitModel(Base):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = Base.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group('models.cnn_lstm')
        parser.add_argument('--cnn-params', type=json.loads, 
                            default=[[3, 24, 5, 2, 2],
                                     [24, 36, 5, 2, 2],
                                     [36, 48, 3, 2, 1],
                                     [48, 64, 3, 1, 1],
                                     [64, 64, 3, 1, 1]])
        parser.add_argument('--cnn-dropout', type=float, default=0.3)
        parser.add_argument('--lstm-size', type=int, default=64)
        parser.add_argument('--q-mlp-params', type=json.loads,
                            default=[32, 32, 2])
        parser.add_argument('--q-mlp-dropout', type=float, default=0.3)

        return parent_parser

    def custom_setup(self):
        assert len(self.hparams.output_mode) == 2, 'CNN-LSTM has 2-dimensional output'
        self._validify_loss_ceof()

        self._cnn = build_cnn(filters=self.hparams.cnn_params,
                              dropout=self.hparams.cnn_dropout)

        cnn_feat_size = self.hparams.cnn_params[-1][0]
        self._lstm = nn.LSTM(cnn_feat_size, self.hparams.lstm_size, batch_first=True)

        q_mlp_params = [self.hparams.lstm_size] + self.hparams.q_mlp_params
        self._q_mlp = build_mlp(filters=q_mlp_params,
                                dropout=self.hparams.q_mlp_dropout,
                                no_act_last_layer=True)

    def forward(self, batch, rnn_state, **kwargs):
        img_seq = batch[0]
        b, t, c, h, w = img_seq.shape
        img_seq_fl = img_seq.view(b * t, c, h, w)
        z = self._cnn(img_seq_fl)
        z = z.max(-1)[0].max(-1)[0] # max-pooling over h and w dims
        z = z.view(b, t, -1)
        
        z, rnn_state = self._lstm(z, rnn_state)
        
        z = z.reshape(b * t, -1)
        z = self._q_mlp(z)
        out = z.view(b, t, -1)

        return out, rnn_state
    
    def get_initial_state(self, batch_size):
        state_size = (1, batch_size, self._lstm.hidden_size) # num_layer x bsize x hidden size
        h_state = torch.zeros(state_size, dtype=torch.float32).to(self.device)
        c_state = torch.zeros(state_size, dtype=torch.float32).to(self.device)
        return [h_state, c_state]
