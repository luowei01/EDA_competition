import numpy as np
import torch
import torch.nn.functional as F
from parse import Parser,Mos
"""求解目标max(V)及对应的s"""

def pair(mos_list):
    mos_pairs = {}
    for mos in mos_list:
        if mos.mid in mos_pairs:
            mos_pairs[mos.mid].append(mos)
        else:
            mos_pairs[mos.mid]=[mos]
    pairs = []
    for moses in mos_pairs.values():
        for i in range(0, len(moses), 2):
            if i+1 < len(moses):
                pairs.append([moses[i], moses[i+1]])
            else:
                pairs.append([moses[i],Mos(0,0,0,0,0,0,0)])
    return pairs
def encode(mos_pairs,words_list):
    encode_dict = {}
    words_list = np.array(words_list)
    encode_dict['name'] = dict(zip(words_list[:,0],[ i for i in range(1,len(words_list[:,0])+1)]))
    nets = set(sum(words_list[:,1:4].tolist(),[]))
    encode_dict['net'] = dict(zip(nets,[i for i in range(1,len(nets)+1)]))
    mos_pairs_encode=[]#7维(name,left, mid, right, type, w, l)的n*2个晶体管
    for item in ['name','left', 'mid', 'right', 'type', 'w', 'l',]:
        temp = []
        for pair in mos_pairs:
            temp1=[]
            for mos in pair:
                if mos.name == 0:
                    code_number = 0
                elif item =='name':
                    code_number = encode_dict['name'][mos.name]
                elif item=='left':
                    code_number = encode_dict['net'][mos.left]
                elif item=='mid':
                    code_number = encode_dict['net'][mos.mid]
                elif item=='right':
                    code_number = encode_dict['net'][mos.right]
                elif item=='type':
                    code_number = 0 if mos.type=='VDD' else 1
                elif item=='w':
                    code_number = mos.w
                elif item=='l':
                    code_number = mos.l
                temp1.append(code_number)
            temp.append(temp1)
        mos_pairs_encode.append(temp)
    # for pair in mos_pairs:
    #     temp = []
    #     for mos in pair:
    #         if mos.name==0:
    #             temp1=[0,0,0,0,0,0,0]
    #         else:
    #             temp1 = [
    #                 encode_dict['name'][mos.name], 
    #                 encode_dict['net'][mos.left], 
    #                 encode_dict['net'][mos.mid],
    #                 encode_dict['net'][mos.right],
    #                 0 if mos.type=='VDD' else 1, 
    #                 mos.w, 
    #                 mos.l]
    #         temp.append(temp1)
    #     mos_pairs_encode.append(temp)
    return mos_pairs_encode


def v_compute(s_old,action):
    """根据赛题规则进行计算新状态的价值"""
    s_new = torch.rand((7,6, 2),requires_grad=True)
    return s_new,torch.sum(s)+torch.sum(action)
    # raise NotImplementedError


def update_s(s):
    s = torch.rand((7,6, 2),requires_grad=True)
    return s


class PolicyNet(torch.nn.Module):
    """策略网络,输入s,输出s下动作概率的分布
       输入:晶体管对的name、left、mid、right、type、w、l
       输出:method 1:任选一个管对，移动该管对到新位置，输出每个管对需要被移动的概率
            method 2:任选两个管对交换位置,输出每个管对需要被交换的概率分布
            method 3:任选一个管对,改变该管对中一个或一对管子的放置方向(左右翻 转 180 度,每个管子是否需要翻转的概率分布

    """

    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(7, 32, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv2d(64,5, kernel_size=2, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # return F.softmax(x, dim=1)
        return x
paser = Parser()
mos_list,words_list = paser.parse('test1.nets')
mos_pairs =pair(mos_list)
s_mos_pairs = torch.tensor(encode(mos_pairs,words_list),requires_grad=True) 
print(s_mos_pairs.shape)
s = s_mos_pairs
"""train"""
answer = np.zeros([2, 12, 3])  
model = PolicyNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 使用Adam优化器
optimizer.zero_grad()
loss_sum = 0
for iter in range(10000):
    optimizer.zero_grad()#置零
    action = model(s)
    s_new,reward = v_compute(s,action)
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
    s = torch.rand(7, 6, 2)
    ac = model(s)
    print(torch.sum(ac)+torch.sum(s))