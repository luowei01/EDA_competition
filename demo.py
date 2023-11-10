'''
Author       : luoweiWHUT 1615108374@qq.com
Date         : 2023-11-07 14:48:53
LastEditors  : luoweiWHUT 1615108374@qq.com
LastEditTime : 2023-11-10 12:00:39
FilePath     : \EDA_competition\demo.py
Description  : 
'''
import torch
import time
import json
import sys
import numpy as np
import random as rd
from enum import Enum
from solver import encode, decode, get_score, v_compute, PolicyNet, selectAndUseOperator, Action_num
from data_parse import Parser


class Algorithm(Enum):
    RD = 0  # 随机算法
    SA = 1  # 退火算法
    RL = 2  # 强化学习
    Roulette = 3  # 轮盘赌


def init_SA():
    ''' 
    description : 计算适合的模拟退火初始参数及部分通用参数
    return       init_state, init_reward, best_state, best_reward, T, T_min,a(降温速度)
    '''
    state = encode(mos_list, encode_dict)
    reward = get_score(state, pins)
    N = 10*len(mos_list)
    count, sum_cost, best_reward = 0, 0, 0
    while count < N:
        action = rd.randint(0, 6)
        new_state = v_compute(state, action)
        new_reward = get_score(new_state, pins_code)
        if new_reward < reward:
            count += 1
            sum_cost += reward-new_reward
        else:
            if best_reward < new_reward:
                best_reward = new_reward
                best_state = new_state
            state = new_state
            reward = new_reward
    return best_state, best_reward, best_state, best_reward, 2*sum_cost/N, 1, 0.97


if __name__ == "__main__":
    start_time = time.time()
    if len(sys.argv) < 3:
        print(
            "ERROR: No enough file provided.\nUsage: python demo.py  <netlist> <cell_name>")
        exit()
    cell_spi_path, test_case_name = sys.argv[1], sys.argv[2]
    paser = Parser()
    mos_list, pins = paser.parse(cell_spi_path, test_case_name)
    encode_dict, decode_dict = paser.build_code_dict(test_case_name)
    pins_code = [encode_dict['net'][net] for net in pins]
    print(f"cell:{test_case_name}\n晶体管数量:{len(mos_list)}")
    use_algorithms = [Algorithm.SA, Algorithm.Roulette]
    print(f"使用{[i.name for i in use_algorithms]}算法优化...")
    """初始化"""
    if Algorithm.RL in use_algorithms:
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = torch.load("model.pt").to(device)
    if Algorithm.Roulette in use_algorithms:
        Weight = [1 for i in range(Action_num)]  # 算子的初始权重，
        UseTimes = [0 for i in range(Action_num)]  # 初始次数，0
        Score = [1 for i in range(Action_num)]  # 算子初始得分，1
        b = 0.5  # 权重变化系数
    state, reward, best_state, best_reward, T, T_min, a = init_SA()  # 利用随机法进行参数初始化
    if Algorithm.SA in use_algorithms:
        if reward > 80:  # 初解得分过高易陷入局部最优
            state = encode(mos_list, encode_dict)
            reward = get_score(state, pins)
        print(f"模拟退火初始温度：{T}")
    else:
        state = encode(mos_list, encode_dict)
        reward = get_score(state, pins)
    print(f"初始布局评分:")
    # evaluator_case('test_case/test.json', test_case_name, cell_spi_path)
    get_score(state, pins_code, True)
    """优化布局"""
    t = 0  # 迭代次数
    while T > T_min:
        count = 0
        while count < 10*len(mos_list):
            t += 1
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
            new_state = v_compute(state, action)
            # 计算新解的价值
            new_reward = get_score(new_state, pins)
            # 更新
            if Algorithm.SA in use_algorithms:
                if Algorithm.Roulette in use_algorithms:
                    if new_reward > reward:  # 新解更优
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
                        if rd.random() < np.exp((new_reward - reward) / T):  # 使用模拟退火算法的接受准则在一定标准下接受劣解
                            state = new_state
                            reward = new_reward
                            Score[action] += 0.8  # 满足接受准则的劣解，权重增加0.8
                        else:
                            Score[action] += 0.6  # 不满足接受准则权重仅增加0.6
                    # 更新算子权重，（1-b）应该放前面，这个例子里取b=0.5，无影响
                    Weight[action] = Weight[action] * b + \
                        (1 - b) * (Score[action] / UseTimes[action])
                else:
                    if new_reward > reward:  # 新解更优
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
                if rd.random() < 0.5:
                    state = new_state
                    reward = new_reward
            count += 1
        T = a*T
    runtime = round(time.time()-start_time, 2)
    # 输出结果
    print(f"迭代次数: {t}\n耗时  : "+"%.2fs" % (runtime))
    print(f"优化后得分:")
    # evaluator_case('best.json', test_case_name, cell_spi_path, runtime)
    get_score(best_state, pins_code, True, runtime)
    with open(f'./output/{test_case_name}.json', 'w') as f:
        json.dump(decode(best_state, decode_dict),
                  f, sort_keys=False, indent=4)
    print(f"布局结果已保存至 ./output/{test_case_name}.json\n"+"*"*150)
