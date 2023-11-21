'''
Author       : luoweiWHUT 1615108374@qq.com
Date         : 2023-11-07 16:17:42
LastEditors  : luoweiWHUT 1615108374@qq.com
LastEditTime : 2023-11-21 17:08:56
FilePath     : \EDA_competition\train.py
Description  : 
'''
import torch
import json
import numpy as np
from tqdm import tqdm
from data_parse import Parser
from solver import encode, decode, REINFORCE, sol_update, get_score, Action_num
from data_parse import Parser
cell_spi_path, cell_name = "public/cells.spi", "SNDSRNQV4"
paser = Parser()
mos_list, pins = paser.parse(cell_spi_path, cell_name)
encode_dict, decode_dict = paser.build_code_dict(cell_name)
pins_code = [encode_dict['net'][net] for net in pins]
ref_width = paser.cell_ref_width_dict[cell_name]
if __name__ == "__main__":
    learning_rate = 1e-3
    num_episodes = 1000
    gamma = 0.98
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"晶体管数量:{len(mos_list)}\n"+"*"*100)
    state_dim = 6
    action_dim = Action_num
    agent = REINFORCE(state_dim, action_dim, learning_rate, gamma, device)
    agent.policy_net = torch.load('model.pt')
    state = encode(mos_list, encode_dict)
    """测试网络输入输出"""
    state_tensor = torch.tensor([state],
                                dtype=torch.float32, device=device)
    print(
        f"测试网络输入输出：\n  input:{state_tensor.shape}\n  out:{agent.policy_net(state_tensor)}\n"+"*"*100)
    print(f"初始布局评分:")
    reward = get_score(state, pins_code, True)
    print("*"*100)
    """train"""
    best_reward = 0
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0  # 回报的均值
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state = encode(mos_list, encode_dict)
                count = 0
                while count < 10*len(mos_list):
                    # while reward < 90:
                    action = agent.take_action(state)
                    # next_state, reward, done, _ = env.step(action)
                    next_state = sol_update(state, action)
                    reward = get_score(next_state, pins_code, ref_width)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    # transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += (reward-episode_return)/(count+1)
                    # if reward > best_reward:
                    #     # with open('best.json', 'w') as f:
                    #     #     json.dump(decode(state, decode_dict),
                    #     #               f, sort_keys=False, indent=4)
                    #     best_state = state
                    #     best_reward = reward
                    count += 1
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' %
                            np.mean(return_list[-10:])
                    })
                pbar.update(1)
    torch.save(agent.policy_net, 'model.pt')
    """"""
