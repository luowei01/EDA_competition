'''
Author       : luoweiWHUT 1615108374@qq.com
Date         : 2023-11-07 14:48:53
LastEditors  : luoweiWHUT 1615108374@qq.com
LastEditTime : 2023-11-22 17:19:11
FilePath     : \EDA_competition\main.py
Description  : 
'''
# import torch
import json
import sys
import copy
import time
import numpy as np
import random as rd
from solver import *
from data_parse import Parser
from enum import Enum
from my_evaluator import evaluator_case


class Algorithm(Enum):
    RD = 0  # 随机算法优化迭代
    SA = 1  # 退火算法优化迭代
    RL = 2  # 强化学习控制新解的生成
    Roulette = 3  # 轮盘赌控制新解的生成
    UseCplus = 4  # 调用c++版本的动态库运行模拟退火算法

def init_SA(state, reward, best_state, best_reward):
    ''' 
    description : 计算适合的模拟退火初始参数及部分通用参数
    return       init_state, init_reward, T, T_min,a(降温速度)
    '''
    # state = encode(mos_list, encode_dict)
    # reward = get_score(state, pins)
    N = 10*len(mos_list)
    count, sum_cost = 0, 0
    while count < N:
        action = rd.randint(0, Action_num-1)
        new_state = sol_update(state, action)
        new_reward = get_score(new_state, pins_code, ref_width)
        if new_reward < reward:
            count += 1
            sum_cost += reward-new_reward
        else:
            if best_reward < new_reward:
                best_state = new_state
                best_reward = new_reward
            state = new_state
            reward = new_reward
    return state, reward, best_state, best_reward, 2*sum_cost/N, 2*sum_cost/N/100


def use_python_run_SA():
    best_state = copy.deepcopy(state)
    best_reward = copy.deepcopy(reward)
    state, reward, best_state, best_reward, T, T_min = init_SA(state, reward, best_state, best_reward)  # 利用爬山法进行参数初始化
    while T > T_min:
        if best_reward > 89.9:
            break
        count = 0
        while count < 20*len(mos_list):
            # 产生新解
            if Algorithm.RL in use_algorithms:
                state_tensor = torch.tensor(
                    [state], dtype=torch.float).to(device)
                probs = model(state_tensor)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample().item()
            elif Algorithm.Roulette in use_algorithms:
                action = selectAndUseOperator(
                    Weight, state, UseTimes)
            else:
                action = rd.randint(0, Action_num-1)
            new_state = sol_update(state, action)
            # 计算新解的价值
            new_reward = get_score(new_state, pins, ref_width)
            # 更新
            if Algorithm.SA in use_algorithms:
                if Algorithm.Roulette in use_algorithms:
                    if new_reward >= reward:  # 新解更优
                        state = new_state
                        reward = new_reward
                        if new_reward > best_reward:
                            # with open('best.json', 'w') as f:
                            #     json.dump(decode(new_state, decode_dict),
                            #               f, sort_keys=False, indent=4)
                            best_state = new_state
                            best_reward = new_reward
                            Score[action] += 1.5  # 如果是最优解的话权重增加到1.5
                    else:
                        # if rd.random() < np.exp((new_reward - reward) / T):  # 使用模拟退火算法的接受准则在一定标准下接受劣解
                        if rd.random() < np.exp((new_reward - reward) / T):
                            state = new_state
                            reward = new_reward
                            Score[action] += 0.8  # 满足接受准则的劣解，权重增加0.8
                        else:
                            Score[action] += 0.6  # 不满足接受准则权重仅增加0.6
                    # 更新算子权重，（1-b）应该放前面，这个例子里取b=0.5，无影响
                    Weight[action] = Weight[action] * b + \
                        (1 - b) * (Score[action] / UseTimes[action])
                else:
                    if new_reward >= reward:  # 新解更优
                        state = new_state
                        reward = new_reward
                        if new_reward > best_reward:
                            # with open('best.json', 'w') as f:
                            #     json.dump(decode(new_state, decode_dict),
                            #               f, sort_keys=False, indent=4)
                            best_state = new_state
                            best_reward = new_reward
                    else:
                        if rd.random() < np.exp((new_reward - reward) / T):  # 使用模拟退火算法的接受准则在一定标准下接受劣解
                            state = new_state
                            reward = new_reward
            else:
                if Algorithm.Roulette in use_algorithms:
                    if new_reward >= reward:  # 新解更优
                        state = new_state
                        reward = new_reward
                        if new_reward > best_reward:
                            # with open('best.json', 'w') as f:
                            #     json.dump(decode(new_state, decode_dict),
                            #               f, sort_keys=False, indent=4)
                            best_state = new_state
                            best_reward = new_reward
                    # 更新算子权重，（1-b）应该放前面，这个例子里取b=0.5，无影响
                    Weight[action] = Weight[action] * b + \
                        (1 - b) * (Score[action] / UseTimes[action])
                else:
                    if new_reward >= reward:  # 新解更优
                        state = new_state
                        reward = new_reward
                        if new_reward > best_reward:
                            # with open('best.json', 'w') as f:
                            #     json.dump(decode(new_state, decode_dict),
                            #               f, sort_keys=False, indent=4)
                            best_state = new_state
                            best_reward = new_reward
                    # else:
                    #     if rd.random() < np.exp((new_reward - reward) / T):  # 使用模拟退火算法的接受准则在一定标准下接受劣解
                    #         state = new_state
                    #         reward = new_reward
            count += 1
        T = a*T
        return best_state


if __name__ == "__main__":
    start = time.time()
    """定义使用的算法"""
    use_algorithms = [Algorithm.UseCplus]
    """读入并解析文件"""
    if len(sys.argv) < 7:
        print(
            "ERROR: No enough file provided.\nUsage: python main.py  -n <netlist> -c <cell_name> -o <save_path>")
        exit()
    for i in range(len(sys.argv)):
        if sys.argv[i] == '-n':
            cell_spi_path = sys.argv[i+1]
        elif sys.argv[i] == '-c':
            cell_name = sys.argv[i+1]
        elif sys.argv[i] == '-o':
            save_path = sys.argv[i+1]
        else:
            continue
    paser = Parser()
    mos_list, pins = paser.parse(cell_spi_path, cell_name)
    encode_dict, decode_dict = paser.build_code_dict(cell_name)
    pins_code = [encode_dict['net'][net] for net in pins]
    ref_width = paser.cell_ref_width_dict[cell_name]
    print(f"cell:{cell_name}\n晶体管数量:{len(mos_list)}")
    print(f"使用{[i.name for i in use_algorithms]}算法优化...")
    sys.stdout.flush()
    """初始化,定义相关算法参数"""
    if Algorithm.RL in use_algorithms:
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = torch.load("model.pt").to(device)
    if Algorithm.Roulette in use_algorithms:
        Weight = [1 for i in range(Action_num)]  # 算子的初始权重，
        UseTimes = [0 for i in range(Action_num)]  # 初始次数，0
        Score = [1 for i in range(Action_num)]  # 算子初始得分，1
        b = 0.5  # 权重变化系数
    if Algorithm.SA in use_algorithms:
        a = 0.95  # a(降温速度)
    """使用优化算法优化布局"""
    # 初始解状态及得分
    state = encode(mos_list, encode_dict)
    reward = get_score(state, pins, ref_width)
    if Algorithm.UseCplus in use_algorithms:
        best_state = use_cplus_run_SA(init_state=state, pinsCode=pins_code, ref_width=ref_width)
    else:
        best_state = use_python_run_SA()
    """保存并对生成的最优解进行评估"""
    with open(save_path, 'w') as f:
        json.dump(decode(best_state, decode_dict),
                  f, sort_keys=False, indent=4)
    evaluator_case(save_path, cell_name, cell_spi_path)
    print(f"耗时:{time.time()-start}s\n"+'*'*150)
    sys.stdout.flush()
