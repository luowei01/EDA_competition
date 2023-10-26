'''
Author       : luoweiWHUT 1615108374@qq.com
Date         : 2023-10-18 18:42:38
LastEditors  : luoweiWHUT 1615108374@qq.com
LastEditTime : 2023-10-26 12:17:07
FilePath     : \EDA_competition\visualize.py
Description  : 
'''
import cv2,numpy
from enum import Enum
class ChannelType(Enum):
    NMOS = 0,
    PMOS = 1
class Mos:
    __slots__=('name','x', 'y', 'source', 'drain', 'width')
    def __init__(self, name,x, y, source, drain, width):
        self.name=name
        self.x = x
        self.y = y
        self.source = source
        self.drain = drain
        self.width = width

    def __lt__(self,other):
        if self.y==other.y:
            return self.x<other.y
        else:
            return self.y > other.y
    def __str__(self):
        return f"'name':{self.name},'x':{self.x}, 'y':{self.y}, 'source':{self.source}, 'drain':{self.drain}, 'width:{self.width}'"
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
    mos_list = []
    for name,mos in  placement_data['placement'].items():
        mos_list.append(Mos(name,int(mos['x']),int(mos['y']),mos['source'], mos['drain'], int(mos['width'])))
    print(mos_list[0])
    mos_list.sort()
    ###visulize
    mos_lenth = 60
    unit = 50
    last_mos = None
    for mos in mos_list:
        if last_mos and last_mos.y == mos.y:
            real_x+= (unit if last_mos.drain==mos.source else 2*unit)        
        else:
            real_x= unit
        rect = ((real_x,110+mos.width//2 if mos.y==1 else 550-mos.width//2),(mos_lenth,mos.width),0)
        box = numpy.int32(cv2.boxPoints(rect))
        cv2.drawContours(result, [box], -1, (0, 255,255), 1)
        rect = ((real_x,330),(5,440),0)
        box = numpy.int32(cv2.boxPoints(rect))
        cv2.drawContours(result, [box], -1, (0, 255,0), -1)
        text_size, _ = cv2.getTextSize(mos.name, None, 0.3, 1)
        cv2.putText(result, mos.name, (real_x-text_size[0]//2,100+text_size[1]//2 if mos.y==1 else 560+text_size[1]//2), None, 0.4, (255,255,255), thickness=1, lineType=cv2.LINE_8, bottomLeftOrigin=None)
        last_mos = mos
    cv2.imwrite('result.jpg',result)

    