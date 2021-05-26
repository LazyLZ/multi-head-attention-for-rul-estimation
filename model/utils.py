from torch import nn
RNN_MAP= {
    'rnn': nn.RNN,
    'lstm': nn.LSTM,
    'gru': nn.GRU
}
ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'gelu': nn.GELU
}