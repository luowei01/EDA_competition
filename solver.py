import numpy as np
import torch
import torch.nn.functional as F
"""求解目标max(V)及对应的s"""


def v_compute(s):
    """根据赛题规则进行计算,价值函数"""
    # return value
    raise NotImplementedError


class PolicyNet(torch.nn.Module):
    """策略网络,输入s,输出s下动作概率的分布
       输入:两行晶体管的宽度、left、mid、right
       输出:两行晶体管的是否需要交换位置、移动方向、是否翻转
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

s = np.zeros([2, 50, 4], np.int8)  # 输入状态:两行晶体管的宽度、left、mid、right
answer = np.zeros([2, 50, 2])  # 输出结果：两行晶体管的位置、是否翻转
v = v_compute(s)
ghgg 