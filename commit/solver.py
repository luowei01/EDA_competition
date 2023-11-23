'''
Author       : luoweiWHUT 1615108374@qq.com
Date         : 2023-11-23 12:16:58
LastEditors  : luoweiWHUT 1615108374@qq.com
LastEditTime : 2023-11-23 13:08:24
FilePath     : \EDA_competition\commit\solver.py
Description  : 定义算法的目标函数、邻域解生成方法、解的编解码方法
'''


import numpy as np
import copy
import random as rd
import networkx as nx


class EulerGraph:

    def __init__(self, refs):
        self.refs = refs
        self.graph = nx.MultiGraph()
        self.build_graph()

    def build_graph(self):
        for r in self.refs:
            if r[0]:
                self.graph.add_node(r[3])
                self.graph.add_node(r[5])
                self.graph.add_edge(r[3], r[5])

    def get_odd_num(self):
        odd_nodes = [node for node,
                     degree in self.graph.degree if degree % 2 != 0]
        return len(odd_nodes)


def sol_update(s_old, action):
    """ 根据action选择更新解的方式
        method 0:任选一个管对，移动该管对到新位置
        method 1:任选两个管对交换位置,输出每个管对需要被交换
        method 2:任选一个管对,改变该管对中一个或一对管子的放置方向
        ...
        return new_sol
    """
    # 创建一个新的的张量数组，用于存储交换后的结果
    s_new = copy.deepcopy(s_old)
    if action == 0:  # 随机选择一个管对，移动到新位置
        a = rd.randint(0, len(s_new[0])-1)
        b = rd.randint(0, len(s_new[0])-1)
        s_new[0].insert(b, s_new[0].pop(a))
        s_new[1].insert(b, s_new[1].pop(a))
    elif action == 1:  # 随机交换两个管对
        a = rd.randint(0, len(s_new[0])-1)
        b = rd.randint(0, len(s_new[0])-1)
        s_new[0][a], s_new[0][b] = s_new[0][b], s_new[0][a]
        s_new[1][a], s_new[1][b] = s_new[1][b], s_new[1][a]
    elif action == 2:  # 交换栅极相同的管子即重新配对
        channel_type = rd.randint(0, 1)
        a = rd.randint(0, len(s_new[0])-1)
        indexs = [i for i in range(len(
            s_new[channel_type])) if s_new[channel_type][i][3] == s_new[channel_type][a][3]]
        b = rd.choice(indexs)
        s_new[channel_type][a], s_new[channel_type][b] = s_new[channel_type][b], s_new[channel_type][a]
    elif action == 3:  # 随机选取一个管子旋转
        a = rd.randint(0, len(s_new[0])-1)
        channel_type = rd.randint(0, 1)
        s_new[channel_type][a][2], s_new[channel_type][a][4] = s_new[channel_type][a][4], s_new[channel_type][a][2]
    # elif action == 4:  # 交换相邻一处管对
    #     a = rd.randint(0, len(s_new[0])-1)
    #     b = a-1 if a > 0 else 1
    #     s_new[0][a], s_new[0][b] = s_new[0][b], s_new[0][a]
    #     s_new[1][a], s_new[1][b] = s_new[1][b], s_new[1][a]
    return s_new


def get_score(mos_list_encode1, pins_code, ref_width1):
    """
    计算当前解(encode后的解)的最终得分
    """
    mos_list_encode = copy.deepcopy(mos_list_encode1)
    # 计算x坐标并set symmetric
    symmetric = 10
    for i in range(len(mos_list_encode[0])):
        if i == 0:
            unit_x = 0
        else:
            if (mos_list_encode[0][i-1][0]*mos_list_encode[0][i][0] == 0 or mos_list_encode[0][i-1][5] == mos_list_encode[0][i][2]) and (mos_list_encode[1][i-1][0]*mos_list_encode[1][i][0] == 0 or mos_list_encode[1][i-1][5] == mos_list_encode[1][i][2]):
                unit_x += 1
            else:
                unit_x += 2
        mos_list_encode[0][i].insert(1, unit_x)
        mos_list_encode[1][i].insert(1, unit_x)
        if mos_list_encode[0][i][0]*mos_list_encode[1][i][0] == 0 and mos_list_encode[0][i][0]+mos_list_encode[1][i][0] > 0:
            symmetric -= 1
    # set drc
    drc = 10
    for i in range(len(mos_list_encode[0])):
        if 0 < i < len(mos_list_encode[0])-1:
            if mos_list_encode[0][i+1][1] - mos_list_encode[0][i-1][1] == 2 and mos_list_encode[0][i-1][0] and mos_list_encode[0][i][0] and mos_list_encode[0][i+1][0] and mos_list_encode[0][i][-1] < mos_list_encode[0][i-1][-1] and mos_list_encode[0][i][-1] < mos_list_encode[0][i+1][-1]:
                drc -= 10
            if mos_list_encode[1][i+1][1] - mos_list_encode[1][i-1][1] == 2 and mos_list_encode[1][i-1][0] and mos_list_encode[1][i][0] and mos_list_encode[1][i+1][0] and mos_list_encode[1][i][-1] < mos_list_encode[1][i-1][-1] and mos_list_encode[1][i][-1] < mos_list_encode[1][i+1][-1]:
                drc -= 10
    # set width
    width = unit_x+1
    # set ref_width
    ref_width = copy.deepcopy(ref_width1)
    mos_list = sum(mos_list_encode, [])
    net_persions = {}
    for mos in mos_list:
        for net, posion in zip(mos[3:6], [mos[1]-0.5, mos[1], mos[1]+0.5]):
            if net in net_persions:
                net_persions[net].append(posion)
            else:
                net_persions[net] = [posion]
    upper_graph = EulerGraph(mos_list_encode[1])
    lower_graph = EulerGraph(mos_list_encode[0])
    min_gap = max(0.0, (upper_graph.get_odd_num() +
                  lower_graph.get_odd_num() - 4) / 2)
    ref_width = (min_gap + ref_width) / 2
    # set pin_access
    pin_coords = []
    for net, r in net_persions.items():
        r.sort()
        if net in pins_code:
            pin_coords.append(r[0])
            max_distance = 0
            for pos in r:
                distance = 0
                another_pos = []
                for another_n, another_r in net_persions.items():
                    if another_n in pins_code and another_n != net:
                        another_pos.extend(another_r)
                if not another_pos:
                    break
                another_pos.sort()
                if another_pos[0] > pos:
                    distance = abs(another_pos[0] - pos)
                elif another_pos[-1] < pos:
                    distance = abs(another_pos[-1] - pos)
                else:
                    for i in range(0, len(another_pos) - 1):
                        if another_pos[i] < pos < another_pos[i + 1]:
                            distance = min(
                                abs(another_pos[i] - pos), abs(another_pos[i + 1] - pos))
                            break
                if distance > max_distance:
                    max_distance = distance
                    pin_coords[-1] = pos
    pin_coords.sort()
    if not pin_coords or len(pin_coords) == 1:
        pin_access = 1
    else:
        pin_spacing = []
        left_spacing = pin_coords[0] + 0.5
        right_spacing = width - 0.5 - pin_coords[-1]
        if left_spacing > 1:
            pin_spacing.append(left_spacing / width)
        if right_spacing > 1:
            pin_spacing.append(right_spacing / width)
        for i in range(0, len(pin_coords) - 1):
            pin_spacing.append(
                (pin_coords[i + 1] - pin_coords[i]) / width)
        pin_access = np.std(np.array(pin_spacing))
    # set bbox
    bbox = 0
    if 0 in net_persions:  # VSS,VDD不计算线长
        del net_persions[0]  # VSS
    if 1 in net_persions:
        del net_persions[1]  # VDD
    for net, persions in net_persions.items():
        bbox += persions[-1]-persions[0]

    # set score
    ws = 40 * (1 - (width - ref_width) / (ref_width + 20))
    bs = min(20.0, 20 * (1 - (bbox - ref_width * (len(pins_code) - 1)) / 60))
    ps = 10 * (1 - pin_access)
    # rs = 10 * (1 / (1 + math.exp(runtime / 3600 - 1)))
    # rs = 7.3
    score = ws+bs+symmetric+drc+ps
    return score


def encode(mos_list, encode_dict):
    """
    将字符串类型的mos管列表编码为int类型
    """
    mos_list_encode = {}  # 6维(name,y,left, mid, right,w)的n个晶体管
    for mos in mos_list:
        temp = []
        # for item in ['name','left', 'mid', 'right', 'type', 'w', 'l',]:
        for item in ['name', 'type', 'left', 'mid', 'right', 'w']:
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
                code_number = 1 if mos.type == 'pch_mac' else 0
            elif item == 'w':
                code_number = mos.w
            elif item == 'l':
                code_number = mos.l
            temp.append(code_number)
        if temp[3] in mos_list_encode:
            mos_list_encode[temp[3]].append(temp)
        else:
            mos_list_encode[temp[3]] = [temp]
    pmos_list = []
    nmos_list = []
    for mid, temp in mos_list_encode.items():
        for mos in temp:
            if mos[1] == 1:
                pmos_list.append(mos)
            else:
                nmos_list.append(mos)
        dif = len(pmos_list) - len(nmos_list)
        if dif == 0:
            continue
        elif dif > 0:
            for i in range(dif):
                nmos_list.append([0, 0] + pmos_list[-1][2:])
        else:
            for i in range(abs(dif)):
                pmos_list.append([0, 1] + nmos_list[-1][2:])
    return [pmos_list, nmos_list]


def decode(mos_list_encode1, decode_dict):
    '''
    description : 将编码后的解还原成可以人工理解的json格式字典
    param        mos_list
    param        {*} mos_list_encode 6维(name,y,left, mid, right,w)的n个晶体管
    param        {*} decode_dict
    return       mos_list_decode #{"placement": {
                                                "M0": {"x": "0","y": "1","source": "VDD","gate": "A2","drain": "ZN","width": "200"},
                                                }
    '''
    mos_list_encode = copy.deepcopy(mos_list_encode1)
    for i in range(len(mos_list_encode[0])):
        if i == 0:
            unit_x = 0
        else:
            if (mos_list_encode[0][i-1][0]*mos_list_encode[0][i][0] == 0 or mos_list_encode[0][i-1][5] == mos_list_encode[0][i][2]) and (mos_list_encode[1][i-1][0]*mos_list_encode[1][i][0] == 0 or mos_list_encode[1][i-1][5] == mos_list_encode[1][i][2]):
                unit_x += 1
            else:
                unit_x += 2
        mos_list_encode[0][i].insert(1, unit_x)
        mos_list_encode[1][i].insert(1, unit_x)
    mos_list = sum(mos_list_encode, [])
    mos_list_decode = {"placement": {}}
    mos_list.sort(key=lambda x: x[1])
    for mos in mos_list:
        if mos[0] == 0:
            continue
        else:
            mos_list_decode['placement'][decode_dict['name'][mos[0]][1:]] = dict(zip(["x", "y", "source", "gate", "drain", "width"],
                                                                                     [str(i) for i in mos[1:3]]+list(map(lambda x: decode_dict['net'][x], mos[3:6]))+[str(mos[6])]))
    return mos_list_decode
