from vistec_ser.models.base_model import BaseSliceModel
from vistec_ser.models.layers.rnn import AttentionLSTM
from torch import nn
import torch
import torch.nn.functional as F

class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(n_channels, n_channels//8, 1, 1)
        self.key = nn.Conv1d(n_channels, n_channels//8, 1, 1)
        self.value = nn.Conv1d(n_channels, n_channels, 1, 1)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()

class CSABLSTM(BaseSliceModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        frame = 1000

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=64, kernel_size=5, padding=2),
                nn.LeakyReLU(),
                nn.LayerNorm(normalized_shape=[64, frame]),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.LayerNorm(normalized_shape=[64, frame//2]),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.LayerNorm(normalized_shape=[128, frame//4]),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.LayerNorm(normalized_shape=[128, frame//8]),
                nn.MaxPool1d(kernel_size=2)
        )

        self.self_attention = SelfAttention(128)

        self.lstm = AttentionLSTM(
            input_dim=128, hidden_dim=128, bidirectional=True, output_dim=4)
        
    def forward(self, x):
        y = self.conv(x)
        o = self.self_attention(y)
        lstm_in = o.transpose(1,2)
        output = self.lstm(lstm_in)
        return output