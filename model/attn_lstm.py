from torch import nn
import torch
from model.utils import ACTIVATION_MAP, RNN_MAP


class MultiHeadAttentionLSTM(nn.Module):
    def __init__(self, cell, sequence_len, feature_num, hidden_dim,
                 fc_layer_dim, rnn_num_layers, output_dim, fc_activation,
                 attention_order, feature_head_num=None, sequence_head_num=None,
                 fc_dropout=0, rnn_dropout=0, bidirectional=False, return_attention_weights=False):
        super().__init__()
        assert cell in ['rnn', 'lstm', 'gru']
        assert fc_activation in ['tanh', 'gelu', 'relu']
        assert isinstance(attention_order, list)

        self.feature_num = feature_num
        self.sequence_len = sequence_len

        self.cell = cell
        self.rnn_hidden_size = hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.bidirectional = bidirectional

        self.fc_layer_dim = fc_layer_dim
        self.fc_dropout = fc_dropout
        self.fc_activation = fc_activation

        self.output_dim = output_dim
        self.rnn_dropout = rnn_dropout

        self.feature_head_num = feature_head_num
        self.sequence_head_num = sequence_head_num

        self.return_attention_weights = return_attention_weights

        self.attention_layers = nn.ModuleList()

        self.attention_order = []
        for attn_type in attention_order:
            if attn_type == 'feature' and self.feature_head_num > 0:
                self.attention_layers.append(
                    nn.MultiheadAttention(
                        embed_dim=self.sequence_len,
                        num_heads=self.feature_head_num
                    )
                )
                self.attention_order.append('feature')
            if attn_type == 'sequence' and self.sequence_head_num > 0:
                self.attention_layers.append(
                    nn.MultiheadAttention(
                        embed_dim=self.feature_num,
                        num_heads=self.sequence_head_num
                    )
                )
                self.attention_order.append('sequence')

        self.rnn = RNN_MAP[self.cell](
            input_size=self.feature_num,
            hidden_size=self.rnn_hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
            num_layers=self.rnn_num_layers,
            dropout=self.rnn_dropout
        )
        linear_in_size = self.rnn_hidden_size
        if self.bidirectional:
            linear_in_size *= 2
        self.linear = nn.Sequential(
            nn.Linear(linear_in_size, self.fc_layer_dim),
            ACTIVATION_MAP[self.fc_activation](),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.fc_layer_dim, output_dim),
        )

        # nn.init.kaiming_normal_(self.linear[0].weight, mode='fan_in')
        # nn.init.kaiming_normal_(self.linear[3].weight, mode='fan_in')
        if self.cell == 'lstm':
            # hidden_size = self.rnn_hidden_size
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            # nn.init.zeros_(self.rnn.bias_ih_l0)
            # nn.init.ones_(self.rnn.bias_ih_l0[hidden_size:hidden_size * 2])

    def forward(self, x):
        # print(x.shape)
        # Attention
        feature_weights = None
        sequence_weights = None
        for i, module in enumerate(self.attention_layers):
            attn_type = self.attention_order[i]
            if attn_type == 'feature':
                a_in = x.permute(2, 0, 1)
                x, feature_weights = module(a_in, a_in, a_in)
                x = x.permute(1, 2, 0)
            if attn_type == 'sequence':
                a_in = x.permute(1, 0, 2)
                x, sequence_weights = module(a_in, a_in, a_in)
                x = x.permute(1, 0, 2)

        # LSTM/GRU
        x, _ = self.rnn(x)

        # Raw
        x = x.contiguous()
        x = x[:, -1, :]
        x = self.linear(x)

        if self.return_attention_weights:
            return x, feature_weights, sequence_weights
        else:
            return x
