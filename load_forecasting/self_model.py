import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import Config

Config = Config()


class Attention(nn.Module):
    def __init__(self, Config):
        super(Attention, self).__init__()
        self.scaling = (Config.hidden / Config.heads) ** -0.5
        self.q_linear = nn.Linear(Config.hidden, Config.hidden, bias=True)
        self.k_linear = nn.Linear(Config.hidden, Config.hidden, bias=True)
        self.v_linear = nn.Linear(Config.hidden, Config.hidden, bias=True)
        self.norm = nn.LayerNorm(Config.hidden)
        self.feed_forward = nn.Sequential(
            nn.Linear(Config.hidden, Config.hidden, bias=True),
            nn.ReLU(),
            nn.Linear(Config.hidden, Config.hidden, bias=True)
        )

    def forward(self, q, k, v):
        Q = self.q_linear(q).transpose(0, 1)
        K = self.k_linear(k).transpose(0, 1)
        V = self.v_linear(v).transpose(0, 1)
        seq_len, batch_size = Q.shape[0], Q.shape[1]
        Q = Q.reshape(seq_len, batch_size * Config.heads, int(Config.hidden / Config.heads)).transpose(0, 1)
        K = K.reshape(seq_len, batch_size * Config.heads, int(Config.hidden / Config.heads)).transpose(0, 1)
        V = V.reshape(seq_len, batch_size * Config.heads, int(Config.hidden / Config.heads)).transpose(0, 1)
        att_weight = torch.bmm(Q, K.transpose(1, 2)) * self.scaling
        att_weight = F.softmax(att_weight, dim=-1)
        att_output0 = torch.bmm(att_weight, V)
        att_output0 = att_output0.transpose(0, 1).reshape(seq_len, batch_size, Config.hidden).transpose(0, 1)
        attn_output1 = att_output0 + k
        attn_output1 = self.norm(attn_output1)
        attn_output2 = self.feed_forward(attn_output1)
        attn_output3 = attn_output2 + attn_output1
        attn_output3 = self.norm(attn_output3)
        return attn_output3


class Main_feature(nn.Module):
    def __init__(self, Config):
        super(Main_feature, self).__init__()
        self.rnn = nn.LSTM(Config.main_size, Config.hidden, batch_first=True)
        self.att = Attention(Config)
        self.embed = nn.Sequential(
            nn.Linear(Config.hidden, Config.hidden),
            # nn.ReLU()
        )

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.att(x, x, x)
        x = self.embed(x)
        return x


class Date_feature(nn.Module):
    def __init__(self, Config):
        super(Date_feature, self).__init__()
        self.rnn = nn.LSTM(Config.date_size, Config.hidden, batch_first=True)
        self.att = Attention(Config)
        self.embed = nn.Linear(Config.hidden, Config.hidden)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.att(x, x, x)
        x = self.embed(x)
        return x


class MAIN_MODEL(nn.Module):
    def __init__(self, Config):
        super(MAIN_MODEL, self).__init__()
        self.main_feature = Main_feature(Config)
        self.date_feature = Date_feature(Config)
        self.att = Attention(Config)
        self.out_proj_layer = nn.Sequential(
            nn.Linear(Config.hidden, Config.hidden, bias=True),
            nn.ReLU(),
            nn.Linear(Config.hidden, Config.output_dim, bias=True),
        )

    def forward(self, x):
        m = x[:, :, :-4]
        d = x[:, :, -4:]
        m_emp = self.main_feature(m)
        demp = self.date_feature(d)
        x = self.att(demp, m_emp, m_emp)
        x = self.att(x, x, x)
        x = self.out_proj_layer(x[:, -1, :])
        return x
