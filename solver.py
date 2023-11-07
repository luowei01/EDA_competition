import numpy as np
import torch
import json
import copy
import random as rd
import torch.nn.functional as F
from tqdm import tqdm
from parse import Parser, Mos
from public.evaluator import get_score
"""求解目标max(V)及对应的s"""


def encode(mos_list, words_list):
    encode_dict = {}
    words_list = np.array(words_list)
    encode_dict['name'] = dict(
        zip(words_list[:, 0], [i for i in range(1,len(words_list[:, 0])+1)]))#0表示虚拟mos
    nets = set(sum(words_list[:, 1:4].tolist(), []))
    encode_dict['net'] = dict(zip(nets, [i for i in range(len(nets))]))
    mos_list_encode = {}  # 6维(name,y,left, mid, right,w)的n个晶体管
    for mos in mos_list:
        temp = []
        # for item in ['name','left', 'mid', 'right', 'type', 'w', 'l',]:
        for item in ['name','type', 'left', 'mid', 'right', 'w']:
            if mos.name == 0:
                code_number = 0
            elif item == 'name':
                code_number = encode_dict['name'][mos.name]
            elif item == 'left':
                code_number = encode_dict['net'][mos.left]
            elif item == 'mid':
                code_number = encode_dict['net'][mos.mid]
            elif item == 'right':
                code_number = encode_dict['net'][mos.right]
            elif item == 'type':
                code_number = 0 if mos.type == 'VDD' else 1
            elif item == 'w':
                code_number = mos.w
            elif item == 'l':
                code_number = mos.l
            temp.append(code_number)
        if temp[3] in mos_list_encode:
            mos_list_encode[temp[3]].append(temp)
        else:
            mos_list_encode[temp[3]]=[temp]
    pmos_list = []
    nmos_list = []
    for mid,temp in mos_list_encode.items():
        for mos in temp:
            if mos[1]==1:
                pmos_list.append(mos)
            else:
                nmos_list.append(mos)
        dif = len(pmos_list) - len(nmos_list)
        if  dif == 0:
            continue
        elif dif >0:
            for i in range(dif):
                nmos_list.append([0,0] + pmos_list[i][2:])
        else:
            for i in range(abs(dif)):
                pmos_list.append([0,1] + pmos_list[i][2:])
    return [pmos_list,nmos_list]


def decode(mos_list_encode1, words_list):
    '''
    description : 
    param        mos_list
    param        {*} mos_list_encode 6维(name,y,left, mid, right,w)的n个晶体管
    param        {*} words_list 
    return       mos_list_decode #{"placement": {
                                                "M0": {"x": "0","y": "1","source": "VDD","gate": "A2","drain": "ZN","width": "200"},
                                                }
    '''
    mos_list_encode = copy.deepcopy(mos_list_encode1)
    encode_dict = {}
    words_list = np.array(words_list)
    encode_dict['name'] = dict(
        zip(words_list[:, 0], [i for i in range(1,len(words_list[:, 0])+1)]))#0表示虚拟mos
    nets = set(sum(words_list[:, 1:4].tolist(), []))
    encode_dict['net'] = dict(zip(nets, [i for i in range(len(nets))]))

    decode_dict ={}
    decode_dict['net'] = {value: key for key, value in encode_dict['net'].items()}
    decode_dict['name'] = {value: key for key, value in encode_dict['name'].items()}

    for i in range(len(mos_list_encode[0])):
        if i == 0:
            unit_x = 0
        else: 
            left_net = [ mos[5] for mos in [mos_list_encode[0][i-1],mos_list_encode[1][i-1]] if mos[0]>0 ]
            right_net = [ mos[2] for mos in [mos_list_encode[0][i],mos_list_encode[1][i]] if mos[0]>0 ]
            min_index = min(len(right_net),len(left_net))
            unit_x+= (1 if right_net[:min_index] == left_net[:min_index] else 2) 
        mos_list_encode[0][i].insert(1,unit_x)
        mos_list_encode[1][i].insert(1,unit_x)
    mos_list = sum(mos_list_encode,[])
    mos_list_decode = {"placement": {}}
    mos_list.sort(key=lambda x:x[1])
    for mos in mos_list:
        if mos[0] == 0:
            continue
        else:
            mos_list_decode['placement'][decode_dict['name'][mos[0]][1:]] = dict(zip(["x", "y", "source", "gate", "drain","width"],
                                                              [str(i) for i in mos[1:3]]+list(map(lambda x: decode_dict['net'][x], mos[3:6]))+[str(mos[6])]))
    return mos_list_decode


def v_compute(s_old, action):
    """根据赛题规则进行计算新状态的价值
        method 0:任选一个管对，移动该管对到新位置
        method 1:任选两个管对交换位置,输出每个管对需要被交换
        method 2:任选一个管对,改变该管对中一个或一对管子的放置方向
    """
    # 创建一个新的的张量数组，用于存储交换后的结果
    s_new = copy.deepcopy(s_old)
    a=rd.randint(0, len(s_new[0])-1)
    if action == 0:#随机选择一个管对，移动到新位置
        b = rd.randint(0, len(s_new[0])-1)
        s_new[0].insert(b,s_new[0].pop(a))
        s_new[1].insert(b,s_new[1].pop(a))
    # elif action==1:#随机交换两个管对
    #     s_new[0][a],s_new[0][b]=s_new[0][b],s_new[0][a]
    #     s_new[1][a],s_new[1][b]=s_new[1][b],s_new[1][a]
    elif action==1:#随机选取一个管子旋转
        channel_type = rd.randint(0,1)
        s_new[channel_type][a][2],s_new[channel_type][a][4]=s_new[channel_type][a][4],s_new[channel_type][a][2]
        # if a<b:
        #     s_new[0][a][2],s_new[0][a][4]=s_new[0][a][4],s_new[0][a][2]
        # else:
        #     s_new[1][a][2],s_new[1][a][4]=s_new[1][a][4],s_new[1][a][2]
    elif action==2:#交换栅极相同的管子即重新配对
        channel_type = rd.randint(0,1)
        indexs = [i for i in range(len(s_new[channel_type])) if s_new[channel_type][i][3]==s_new[channel_type][a][3]]
        b = rd.choice(indexs)
        s_new[channel_type][a],s_new[channel_type][b]=s_new[channel_type][b],s_new[channel_type][a]

    with open('temp.json', 'w') as f:
        json.dump(decode(s_new, words_list),
                  f, sort_keys=False, indent=4)
    reward = get_score('temp.json', 'SNDSRNQV4', 'test_case/cells.spi')
    return s_new, reward
    # raise NotImplementedError


class IntegerRangeOutputLayer(torch.nn.Module):
    def __init__(self, max_value=10):
        super(IntegerRangeOutputLayer, self).__init__()
        self.max_value = max_value

    def forward(self, x):
        integer_output = torch.round(x)  # 使用四舍五入
        integer_output[:, 0] = torch.clamp(
            integer_output[:, 0], min=0, max=self.max_value)  # 第一列限制在0和max_value之间
        integer_output[:, 1] = torch.clamp(
            integer_output[:, 1], min=0, max=1)  # 第一列限制在0和1之间
        return integer_output


class PolicyNet(torch.nn.Module):
    """策略网络,输入s,输出s下动作概率的分布
       输入:2行晶体管的y、left、mid、right、width
       输出:method 1:任选一个管对，移动该管对到新位置，输出每个管对需要被移动的概率
            method 2:任选两个管对交换位置,输出每个管对需要被交换的概率分布
            method 3:任选一个管对,改变该管对中一个或一对管子的放置方向
    """

    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(state_dim, 64, kernel_size=1)
        self.fc1 = torch.nn.Linear(64*2*30, 128)
        self.fc2 = torch.nn.Linear(128, action_dim)
        # self.integer_output = IntegerRangeOutputLayer()

    def forward(self, x):
        # x = x.permute(2, 0, 1).unsqueeze(0)  # [2,2,5]->[1,5,2,2]
        x = x.permute(0,3, 1, 2)  # [1,2,2,5]->[1,5,2,2]
        desired_shape = (1,6, 2, 30)
        desired_shape = (1,6, 2, 30)
        padded_input = torch.zeros(desired_shape,device="cuda:0")
        padded_input[:,:, :, :x.shape[-1]] = x
        x = padded_input
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # x = self.integer_output(x)  # 使用自定义的整数输出层
        x = F.softmax(self.fc2(x), dim=-1)
        return x


class REINFORCE:
    def __init__(self, state_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降


if __name__ == "__main__":
    rd.seed(1)
    learning_rate = 1e-3
    num_episodes = 1000
    gamma = 0.98
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    paser = Parser()
    mos_list, words_list = paser.parse('test_case/cells.spi')
    print(f"晶体管数量:{len(mos_list)}\n"+"*"*100)
    state_dim = 6
    action_dim = 3
    agent = REINFORCE(state_dim, action_dim, learning_rate, gamma, device)
    """测试网络输入输出"""
    state = torch.tensor([encode(mos_list, words_list)],
                     dtype=torch.float32, device=agent.device, requires_grad=True)
    print(f"测试网络输入输出：\n  input:{state.shape}\n  out:{agent.policy_net(state)}\n"+"*"*100)
    """测试结果输出:test_case/test.json"""
    with open('test_case/test.json', 'w') as f:
        json.dump(decode(state[0].int().tolist(), words_list),
                  f, sort_keys=False, indent=4)
    print(f"测试结果译码:test_case/test.json \n"+"*"*100)
    """测试初始结果得分："""
    reward = get_score('test_case/test.json', 'SNDSRNQV4', 'test_case/cells.spi')
    print(f"初始布局得分：{reward}\n"+"*"*100)
    """train"""
    best_reward = 0
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state = encode(mos_list, words_list)
                count = 0
                while count<10*len(mos_list):
                    if reward == -100:
                        print(f"{state[0]}\n{state[1]}")
                        with open('error.json', 'w') as f:
                            json.dump(decode(state, words_list),
                                      f, sort_keys=False, indent=4)
                        exit()
                    elif reward>80:
                        with open('best.json', 'w') as f:
                            json.dump(decode(state, words_list),
                                      f, sort_keys=False, indent=4)
                    elif reward>best_reward:
                        best_reward = reward
                        with open('best.json', 'w') as f:
                            json.dump(decode(state, words_list),
                                      f, sort_keys=False, indent=4)
                    action = agent.take_action(state)
                    # next_state, reward, done, _ = env.step(action)
                    next_state, reward = v_compute(state, action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    # transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    count+=1
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    torch.save(agent.policy_net, 'model.pt')
    """"""
