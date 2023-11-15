'''
Author       : luoweiWHUT 1615108374@qq.com
Date         : 2023-11-15 16:54:11
LastEditors  : luoweiWHUT 1615108374@qq.com
LastEditTime : 2023-11-15 17:20:44
FilePath     : \EDA_competition\my_evaluator.py
Description  : 
'''
import json
from public.evaluator import *


class Cell(Cell):
    """ Dual row cell.
    """

    def __init__(self, name, pins):
        super(Cell, self).__init__(name, pins)

    def evaluate(self, runtime, return_flag=False):
        for net, r in self.net_pos.items():
            r.sort()
            if not is_power(net):
                self.bbox += r[-1] - r[0]
        self.get_pin_access()

        ws = 40 * (1 - (self.width - self.ref_width) / (self.ref_width + 20))
        bs = min(20.0, 20 * (1 - (self.bbox - self.ref_width * (len(self.pins) - 1)) / 60))
        ps = 10 * (1 - self.pin_access)
        ss = self.symmetric
        ds = self.drc
        rs = 10 * (1 / (1 + math.exp(runtime / 3600 - 1)))
        self.score = ws + bs + ps + ss + ds + rs
        if return_flag:
            return [self.score, ws, bs, ps, self.symmetric, self.drc, rs]
        else:
            print("Cell score %f (width: %d, bbox: %f, pin_access: %f, symmetric: %d, drc: %d, runtime: %ds)"
                  % (self.score, self.width, self.bbox, self.pin_access, self.symmetric, self.drc, runtime))
            print("Cell score %f (width: %d, bbox: %f, pin_access: %f, symmetric: %d, drc: %d, runtime: %d)"
                  % (self.score, ws, bs, ps, self.symmetric, self.drc, rs))


def evaluator_case(placement_file, cell_name, netlist_file, runtime=0):
    # read placement file
    placement_stream = open(placement_file, "r")
    placement_dic = json.load(placement_stream)
    placement = placement_dic["placement"]

    # get transistor properties from netlist
    transistor_dic, pins = load_netlist(netlist_file, cell_name)
    # get ref width
    ref_width = 0
    for transistor_name, t in transistor_dic.items():
        if t.channel_width > 220:
            ref_width += t.channel_width // 200
        else:
            ref_width += 1

    cell = Cell(cell_name, pins)
    # get cell width
    width = 0
    for transistor_name, properties in placement.items():
        if width < int(properties["x"]) + 1:
            width = int(properties["x"]) + 1
    cell.reset(width)
    # add transistor
    for transistor_name, properties in placement.items():
        tname, finger = decompose_transistor_name(transistor_name)
        transistor = transistor_dic[tname]
        ref = TransistorRef(transistor, not str_equal(
            transistor.source_net, properties["source"]), int(properties["width"]))
        cell.add_transistor(ref, int(properties["x"]))

    # set ref width
    upper_graph = EulerGraph(cell.upper)
    lower_graph = EulerGraph(cell.lower)
    min_gap = max(0.0, (upper_graph.get_odd_num() +
                  lower_graph.get_odd_num() - 4) / 2)
    cell.ref_width = (min_gap + ref_width) / 2

    # check and get score
    if cell.check(transistor_dic):
        return cell.evaluate(runtime, return_flag=True)
