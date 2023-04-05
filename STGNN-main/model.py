import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
import math

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


class Transform(nn.Module):
    # Transformer过程
    def __init__(self, outfea, d):
        super(Transform, self).__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)

        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)

        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.ReLU(),
            nn.Linear(outfea, outfea)
        )

        self.d = d

    def forward(self, x):
        query = self.qff(x)
        key = self.kff(x)
        value = self.vff(x)

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0, 2, 3, 1)
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0, 2, 1, 3)

        A = torch.matmul(query, key)
        A /= (self.d ** 0.5)
        A = torch.softmax(A, -1)

        value = torch.matmul(A, value)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0, 2, 1, 3)
        value += x

        value = self.ln(value)
        x = self.ff(value) + value
        return self.lnff(x)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, outfea, max_len=12):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, outfea).to(device)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, outfea, 2) *
                             -(math.log(10000.0) / outfea))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)  # [1,T,1,F]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe,
                         requires_grad=False)
        return x


class SGNN(nn.Module):
    def __init__(self, outfea):
        super(SGNN, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.Linear(outfea, outfea)
        )
        self.ff1 = nn.Linear(outfea, outfea)

    def forward(self, x):
        # 这里的64应该是为每个节点学习一个位置表示，使用一个含有两个全连接层的模块学习
        p = self.ff(x)
        # p.transpose(-1, -2).shape = (16,64,325)
        a = torch.matmul(p, p.transpose(-1, -2))
        # p*p.transpose(-1, -2) 即 (16,325,64) * (16,64,325) = (16,325,325) 得到的(325,325)就是要学习到的位置关系矩阵
        # 这里的a就相当于论文中的Score
        # torch.softmax(a, -1) 对最后一维进行softmax。对于(16,325,325)就是对16个数据的列进行softmax
        # torch.eye(x.shape[1])会生成一个(325,325)的对角矩阵
        R = torch.relu(torch.softmax(a, -1)) + torch.eye(x.shape[1]).to(device)

        # D是R的度矩阵的-1/2次幂
        D = (R.sum(-1) ** -0.5)
        # 处理inf值
        D[torch.isinf(D)] = 0.
        # 因为用R.sum(-1)求度矩阵，因此维度变为一维了，要将度矩阵还原为对角形式
        # 补充知识：度矩阵是一个对角矩阵，(i,i)的值表示和i相关联的边的个数，当然在这里(i,i)的值表示的是一个学习出来的相关度
        D = torch.diag_embed(D)

        # 按照论文中的公式进行图特征提取
        A = torch.matmul(torch.matmul(D, R), D)

        # 我认为可以使用sigmoid
        # x = torch.sigmoid(self.ff1(torch.matmul(A, x)))
        x = torch.relu(self.ff1(torch.matmul(A, x)))

        # 经过一个全连接层得到我们想要的输出shape x=(16,325,64)
        # 这个维度可以理解为为每一个节点用GNN学习了一个64维的embedding

        return x


class GRU(nn.Module):
    def __init__(self, outfea):
        super(GRU, self).__init__()
        self.ff = nn.Linear(2 * outfea, 2 * outfea)
        self.zff = nn.Linear(2 * outfea, outfea)
        self.outfea = outfea

    def forward(self, x, xh):
        r, u = torch.split(torch.sigmoid(self.ff(torch.cat([x, xh], -1))), self.outfea, -1)
        z = torch.tanh(self.zff(torch.cat([x, r * xh], -1)))
        x = u * z + (1 - u) * xh
        return x


class STGNNwithGRU(nn.Module):
    # outfea=64
    def __init__(self, outfea):
        super(STGNNwithGRU, self).__init__()
        self.sgnnh = nn.ModuleList([SGNN(outfea) for i in range(12)])
        self.sgnnx = nn.ModuleList([SGNN(outfea) for i in range(12)])
        self.gru = nn.ModuleList([GRU(outfea) for i in range(12)])

    def forward(self, x):
        B, T, N, F = x.shape
        # B=16,T=12,N=325,F=64
        hidden_state = torch.zeros([B, N, F]).to(device)
        # hidden_state.shape=[16,325,64]
        output = []

        for i in range(T):
            # 因为我们需要使用时序网络以提取时序相关性，因此这里使用了[:,i,:,:]进行提取
            # 这样就将12个时序上相关的数据连接起来
            gx = self.sgnnx[i](x[:, i, :, :])
            # 当i=0的时候，就是刚进入的时候。此时并不存在从上一个层过来的时序表示，因此使用了torch.zeros
            gh = hidden_state
            if i != 0:
                gh = self.sgnnh[i](hidden_state)
            # 除了第一个gru之外，其他的gru都要接受当前时刻的数据和从上一个时刻传来的数据
            hidden_state = self.gru[i](gx, gh)
            # hidden_state的shape是(16,325,64),然后能看到每次的hidden_state会被赋值到gh作为下一次sgnn的输入(每个gru中间是有sgnn的,因此每次的hidden_state还需要通过sgnn才会到达下面的一个gru)
            output.append(hidden_state)
            # 由于输出的是经过gru处理的数据，因此用output记录每个gru输出的数据

        output = torch.stack(output, 1)

        return output


class STGNN(nn.Module):
    # infea=1, outfea=64, L=1, d=16
    def __init__(self, infea, outfea, L, d):
        super(STGNN, self).__init__()
        self.start_emb = nn.Linear(infea, outfea)
        self.end_emb = nn.Linear(outfea, infea)

        self.stgnnwithgru = nn.ModuleList([STGNNwithGRU(outfea) for i in range(L)])
        self.positional_encoding = PositionalEncoding(outfea)
        self.transform = nn.ModuleList([Transform(outfea, d) for i in range(L)])

        self.L = L

    def forward(self, x):
        '''
        input [B,T,N]  (batch, time, node)
        '''
        # 这里我们使用args中设置的参数对模型每一步的shape进行解释，以方便理解
        # 输入x为[16,12,325] 就是一个batch中有12个样本，每个样本中
        x = x.unsqueeze(-1)
        # unsqueeze升维，数据维度变为[16,12,325,1]
        x = self.start_emb(x)
        # embbeing后变为[16,12,325,64]
        # self.L是stgnnwithgru的层数（时空提取的层数）
        for i in range(self.L):
            x = self.stgnnwithgru[i](x)
        x = self.positional_encoding(x)
        for i in range(self.L):
            x = self.transform[i](x)
        # 最后一个预测层
        x = self.end_emb(x)
        # 得到的x.shape(16,12,325,1)
        # return的时候squeeze (-1即可)
        return x.squeeze(-1)
