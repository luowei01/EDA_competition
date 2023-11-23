'''
Author       : luoweiWHUT 1615108374@qq.com
Date         : 2023-11-23 12:16:58
LastEditors  : luoweiWHUT 1615108374@qq.com
LastEditTime : 2023-11-23 13:02:25
FilePath     : \EDA_competition\commit\main.py
Description  :  主程序，模拟退火算法定义
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
from my_evaluator import evaluator_case


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
        action = rd.randint(0, 3)
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
    global state, reward
    best_state = copy.deepcopy(state)
    best_reward = copy.deepcopy(reward)
    state, reward, best_state, best_reward, T, T_min = init_SA(state, reward, best_state, best_reward)  # 利用爬山法进行参数初始化
    while T > T_min:
        if best_reward > 89.9:
            break
        count = 0
        while count < 20*len(mos_list):
            # 产生新解
            action = rd.randint(0, 3)
            new_state = sol_update(state, action)
            # 计算新解的价值
            new_reward = get_score(new_state, pins, ref_width)
            # 更新
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
            count += 1
        T = a*T
    return best_state


if __name__ == "__main__":
    start = time.time()
    """读入文件路径"""
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
    """解析文件"""
    paser = Parser()
    mos_list, pins = paser.parse(cell_spi_path, cell_name)
    encode_dict, decode_dict = paser.build_code_dict(cell_name)
    pins_code = [encode_dict['net'][net] for net in pins]
    ref_width = paser.cell_ref_width_dict[cell_name]
    print(f"cell:{cell_name}\n晶体管数量:{len(mos_list)}")
    """初始化,定义相关算法参数"""
    a = 0.95  # a(降温速度)
    """使用优化算法优化布局"""
    best_sol_list = []
    for i in range(5):
        state = encode(mos_list, encode_dict)
        reward = get_score(state, pins, ref_width)
        best_state = use_python_run_SA()
        best_sol_list.append(best_state)
    best_sol = max(best_sol_list, key=lambda x: get_score(x, pins_code, ref_width))
    """保存并对生成的最优解进行评估"""
    with open(save_path, 'w') as f:
        json.dump(decode(best_sol, decode_dict),
                  f, sort_keys=False, indent=4)
    evaluator_case(save_path, cell_name, cell_spi_path)
    print(f"耗时:{time.time()-start}s\n"+'*'*150)
