'''
Author       : luoweiWHUT 1615108374@qq.com
Date         : 2023-10-12 11:47:36
LastEditors  : luoweiWHUT 1615108374@qq.com
LastEditTime : 2023-11-10 19:56:21
FilePath     : \EDA_competition\data_parse.py
Description  : 
'''
import re
import numpy as np
import copy


class Mos:
    __slots__ = ('name', 'type', 'left', 'mid', 'right', 'w', 'l',)

    def __init__(self, name, left, mid, right, type, w, l):
        self.name = name
        self.left = left
        self.mid = mid
        self.right = right
        self.type = type
        self.w = w
        self.l = l

    def __repr__(self):
        return f"'name':{self.name},'y':{self.type}, 'source':{self.left},'gain':{self.mid}, 'drain':{self.right}, 'width:{self.w}'\n"


class Parser:
    def __init__(self) -> None:
        self.cell_dict = {}
        self.cell_pins_dict = {}
        self.cell_words_dict = {}
        self.cell_encode_dict = {}
        self.cell_decode_dict = {}
        # self.attributes = ('name', 'type', 'left', 'mid', 'right', 'w', 'l')
        self.pattern = re.compile(
            r"(.*)\s(.*)\s(.*)\s(.*)\s(.*)\s.*\s(\w)=(.*\w)\s\w=(.*\w)\s*\n")

    def parse(self, path, cell_name=None):
        self.path = path
        if cell_name:
            if cell_name in self.cell_dict:
                print(f"{cell_name}已存在搜索记录")
                return self.cell_dict[cell_name], self.cell_pins_dict[cell_name]
            with open(self.path, 'r') as f:
                find_cell = False
                for line in f:
                    words = line.split()
                    if words[0] == ".ENDS" and find_cell:
                        break
                    if words[0] == ".SUBCKT" and cell_name == words[1]:
                        find_cell = True
                        self.cell_pins_dict[cell_name] = [
                            net for net in words[2:] if net.upper() not in ["VDD", "VSS"]]
                        self.cell_dict[cell_name] = []
                        self.cell_words_dict[cell_name] = []
                        continue
                    if find_cell:
                        line_data = re.match(self.pattern, line)
                        if line_data:
                            name, left, mid, right, type, swap_wl, w, l = line_data.groups()
                            if swap_wl.upper() == 'L':
                                w, l = l, w
                            params = [name, left, mid, right, type, int(
                                float(w[:-1])*1000) if w[-1] == 'u' else int(float(w[:-1])), int(
                                float(l[:-1])*1000) if l[-1] == 'u' else int(float(l[:-1]))]
                            if params[-2] > 220:
                                if params[-2] % 2 == 0:
                                    params[-2] = int(params[-2]/2)
                                    self.cell_dict[cell_name].append(
                                        Mos(*params))
                                    self.cell_words_dict[cell_name].append(
                                        copy.deepcopy(params))
                                    params[0] += '_finger1'
                                    self.cell_dict[cell_name].append(
                                        Mos(*params))
                                    self.cell_words_dict[cell_name].append(
                                        params)
                                else:
                                    params[-2] = int(params[-2]/2 + 2.5)
                                    self.cell_dict[cell_name].append(
                                        Mos(*params))
                                    self.cell_words_dict[cell_name].append(
                                        copy.deepcopy(params))
                                    params[0] += '_finger1'
                                    params[-2] -= 5
                                    self.cell_dict[cell_name].append(
                                        Mos(*params))
                                    self.cell_words_dict[cell_name].append(
                                        params)
                            else:
                                self.cell_dict[cell_name].append(Mos(*params))
                                self.cell_words_dict[cell_name].append(params)
            if not find_cell:
                print(
                    f"未能成功在 {self.path} 中搜索到名为 {cell_name} 的cell,请重新确认输入参数!\nUsage: python demo.py <netlist> <cell_name>")
                exit()
            return self.cell_dict[cell_name], self.cell_pins_dict[cell_name]
        else:
            with open(self.path, 'r') as f:
                for line in f:
                    words = line.split()
                    if words[0] == ".ENDS":
                        continue
                    if words[0] == ".SUBCKT":
                        cell_name = words[1]
                        self.cell_pins_dict[cell_name] = [
                            net for net in words[2:] if net.upper() not in ["VDD", "VSS"]]
                        self.cell_dict[cell_name] = []
                        self.cell_words_dict[cell_name] = []
                        continue
                    line_data = re.match(self.pattern, line)
                    if line_data:
                        name, left, mid, right, type, swap_wl, w, l = line_data.groups()
                        if swap_wl.upper() == 'L':
                            w, l = l, w
                        params = [name, left, mid, right, type, int(
                            float(w[:-1])*1000) if w[-1] == 'u' else int(float(w[:-1])), int(
                            float(l[:-1])*1000) if l[-1] == 'u' else int(float(l[:-1]))]
                        if params[-2] > 220:
                            if params[-2] % 2 == 0:
                                params[-2] = int(params[-2]/2)
                                self.cell_dict[cell_name].append(Mos(*params))
                                self.cell_words_dict[cell_name].append(
                                    copy.deepcopy(params))
                                params[0] += '_finger1'
                                self.cell_dict[cell_name].append(Mos(*params))
                                self.cell_words_dict[cell_name].append(params)
                            else:
                                params[-2] = int(params[-2]/2 + 2.5)
                                self.cell_dict[cell_name].append(Mos(*params))
                                self.cell_words_dict[cell_name].append(
                                    copy.deepcopy(params))
                                params[0] += '_finger1'
                                params[-2] -= 5
                                self.cell_dict[cell_name].append(Mos(*params))
                                self.cell_words_dict[cell_name].append(params)
                        else:
                            self.cell_dict[cell_name].append(Mos(*params))
                            self.cell_words_dict[cell_name].append(params)
            return self.cell_dict, self.cell_pins_dict

    def build_code_dict(self, cell_name):
        self.cell_encode_dict[cell_name] = {}
        words_list = np.array(self.cell_words_dict[cell_name])
        self.cell_encode_dict[cell_name]['name'] = dict(
            zip(words_list[:, 0], [i for i in range(1, len(words_list[:, 0])+1)]))  # 0表示虚拟mos
        nets = set(sum(words_list[:, 1:4].tolist(), []))
        nets.remove('VDD')
        nets.remove('VSS')
        self.cell_encode_dict[cell_name]['net'] = {'VSS': 0, 'VDD': 1}
        self.cell_encode_dict[cell_name]['net'].update(
            dict(zip(nets, [i for i in range(2, len(nets)+2)])))
        self.cell_decode_dict[cell_name] = {}
        self.cell_decode_dict[cell_name]['net'] = {value: key for key,
                                                   value in self.cell_encode_dict[cell_name]['net'].items()}
        self.cell_decode_dict[cell_name]['name'] = {value: key for key,
                                                    value in self.cell_encode_dict[cell_name]['name'].items()}
        return self.cell_encode_dict[cell_name], self.cell_decode_dict[cell_name]


if __name__ == "__main__":
    cell_spi_path, test_case_name = "public/cells.spi", "SNDSRNQV4"
    paser = Parser()
    # paser.parse(cell_spi_path)
    mos_list, pins = paser.parse(cell_spi_path, test_case_name)
    encode_dict, decode_dict = paser.build_code_dict(test_case_name)
    pins_code = [encode_dict['net'][net] for net in pins]
    print(len(mos_list))
