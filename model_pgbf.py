import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False))


class PGBF(nn.Module):
    def __init__(self,
                 model_size_omic: str = 'small', omic_sizes=[100, 200, 300, 400, 500, 600],
                 dim_in=384, dim_hidden=512, topk=6, dropout=0.3):
        super(PGBF, self).__init__()

        ### Constructing Genomic SNN
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        hidden = self.size_dict_omic[model_size_omic]  # 隐藏层大小
        sig_networks = []  # 存储所有处理基因的网络模块
        for input_dim in omic_sizes:  # omic_sizes=[100, 200, 300, 400, 500, 600]
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]  # 第一层
            for i, _ in enumerate(hidden[1:]):  # 遍历 hidden 中的后续维度，构建层与层之间的连接
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))  # 将该网络添加到 sig_networks
        self.sig_networks = nn.ModuleList(sig_networks)  # 存储多个神经网络模块

        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.LeakyReLU())

        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.scale = dim_hidden ** -0.5

        self.topk = topk

        self.linear1 = nn.Linear(dim_hidden, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)

        self.activation = nn.LeakyReLU()

        self.message_dropout = nn.Dropout(dropout)

        att_net = nn.Sequential(nn.Linear(dim_hidden, dim_hidden // 2), nn.LeakyReLU(), nn.Linear(dim_hidden // 2, 1))
        self.readout = GlobalAttention(att_net)

    def forward(self, x_path, **kwargs):

        # 处理基因数据 形成omic_bag的特征
        # x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)]
        # # each omic signature goes through it's own FC layer
        # h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        # h_omic_bag = torch.stack(h_omic).unsqueeze(1)


        # WiKG 公式1 每个补丁的嵌入投影为头部和尾部嵌入
        x_path = self._fc1(x_path)  # 将维度从 dim_in 转换为 dim_hidden
        x_path = (x_path + x_path.mean(dim=1, keepdim=True)) * 0.5  # 使特征分布更加平滑，有助于训练稳定性
        e_h = self.W_head(x_path)  # embedding_head
        e_t = self.W_tail(x_path)  # embedding_tail

        # WiKG 公式2 3 相似性得分最高的前 k 个补丁被选为补丁 i 的邻居
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 计算 e_h 和 e_t 之间的相似性(点积)
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # 获取 Top - k 注意力分数和对应索引
        topk_prob = F.softmax(topk_weight, dim=2)  # 归一化注意力分数

        topk_index = topk_index.to(torch.long)  # 转换索引类型
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # 扩展索引以匹配 e_t 维度
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # 创建批次索引辅助张量
        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # 使用索引从 e_t 中提取特征向量 neighbors_head

        # WiKG 公式 4 为有向边分配嵌入 embedding head_r
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))
        # WiKG 公式 6 计算 加权因子
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)
        # WiKG 公式 7 对 加权因子 进行归一化
        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        # WiKG 公式 5 计算补丁 i 相邻 N （i） 的尾部嵌入的线性组合
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)
        # WiKG 公式 8 将聚合的邻居信息 e_Nh 与原始 head 融合
        sum_embedding = self.activation(self.linear1(e_h + e_Nh))
        bi_embedding = self.activation(self.linear2(e_h * e_Nh))
        e_h = sum_embedding + bi_embedding
        # WiKG 公式 9 生成 graph-level 嵌入 embedding_graph
        e_h = self.message_dropout(e_h)
        e_g = self.readout(e_h.squeeze(0), batch=None)

        return e_g


if __name__ == "__main__":
    data = torch.randn((1, 10000, 384)).to(device)
    print('input.shape:', data.shape)
    model = PGBF(dim_in=384, dim_hidden=512).to(device)
    output_path = model(data)
    print('output_path.shape:', output_path.shape)
