'''
Author       : luoweiWHUT 1615108374@qq.com
Date         : 2023-10-18 18:42:38
LastEditors  : luoweiWHUT 1615108374@qq.com
LastEditTime : 2023-10-18 20:13:48
FilePath     : \EDA_competition\visualize.py
Description  : 
'''
import cv2,numpy
from enum import Enum
from typing import Any, Tuple
class ChannelType(Enum):
    NMOS = 0,
    PMOS = 1
class Transistor:
    """
    Abstract representation of a MOS transistor.
    """

    def __init__(self, channel_type: ChannelType,
                 source_net: str, gate_net: str, drain_net: str,
                 channel_width, name: str):
        """
        params:
        left: Either source or drain net.
        right: Either source or drain net.
        """
        self.name = name
        self.channel_type = channel_type
        self.source_net = source_net
        self.gate_net = gate_net
        self.drain_net = drain_net
        self.channel_width = channel_width

    def terminals(self) -> Tuple[Any, Any, Any]:
        """ Return a tuple of all terminal names.
        :return:
        """
        return self.source_net, self.gate_net, self.drain_net

    def __key(self):
        return self.name, self.channel_type, self.source_net, self.gate_net, self.drain_net, self.channel_width

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, y):
        return self.__key() == y.__key()
if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) <2:
        print("ERROR: No placement file provided.")
        exit()
    placement_file=sys.argv[1]
    with open(placement_file, 'r') as f:
        placement_data = json.load(f)
    result = numpy.zeros((640,640,3),numpy.uint8)
    for name,mos in  placement_data['placement'].items():
        rect = ((30*int(mos['x'])+30,220*int(mos['y'])+220),(30,int(mos['width'])),0)
        box = numpy.int32(cv2.boxPoints(rect))
        cv2.drawContours(result, [box], -1, (0, 255,255), 1)
        rect = ((30*int(mos['x'])+30,330),(5,440),0)
        box = numpy.int32(cv2.boxPoints(rect))
        cv2.drawContours(result, [box], -1, (0, 255,0), -1)
        cv2.putText(result, name, (30*int(mos['x'])+30,100 if int(mos['y'])==0 else 560), None, 0.3, (255,255,255), thickness=1, lineType=cv2.LINE_8, bottomLeftOrigin=None)
    cv2.imwrite('result.jpg',result)

    