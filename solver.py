import numpy as np
import torch
import torch.nn.functional as F
"""求解目标max(V)及对应的s"""
N = 12


def v_compute(s):
    """根据赛题规则进行计算,价值函数"""
    return torch.sum(s)
    # raise NotImplementedError


def update_s(s):
    s = torch.rand(2, 12, 4)
    s.requires_grad = True
    return s


class PolicyNet(torch.nn.Module):
    """策略网络,输入s,输出s下动作概率的分布
       输入:两行晶体管的宽度、left、mid、right
       输出:method 1:任选一个管对，移动该管对到新位置，输出每个管对需要被移动的概率
            method 2:任选两个管对交换位置,输出每个管对需要被交换的概率分布
            method 3:任选一个管对,改变该管对中一个或一对管子的放置方向(左右翻 转 180 度,每个管子是否需要翻转的概率分布

    """

    def __init__(self, action_dim=3, in_channels=2):
        super(PolicyNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 10, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(10, 5, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv2d(5, 2, kernel_size=2, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return F.softmax(x, dim=1)
        # return x


"""train"""
s = torch.rand(2, 12, 4)  # 输入状态:两行晶体管的宽度、left、mid、right
s.requires_grad = True
answer = np.zeros([2, 12, 3])  # 输出结果：两行晶体管的位置、是否翻转
model = PolicyNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 使用Adam优化器
optimizer.zero_grad()
loss_sum = 0
for iter in range(10000):
    optimizer.zero_grad()#置零
    action = model(s)
    reward = torch.sum(action)+torch.sum(s)
    loss = torch.abs(100-reward)
    loss.backward()
    optimizer.step()  # 梯度下降

    loss_sum += loss
    s = update_s(s)
    if iter % 100 == 0:
        print(loss_sum/100)
        loss_sum = 0
torch.save(model, 'model.pt')
""""""
model = torch.load('model.pt')
for i in range(100):
    s = torch.rand(2, 12, 4)
    ac = model(s)
    print(torch.sum(ac)+torch.sum(s))