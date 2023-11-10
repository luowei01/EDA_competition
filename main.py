'''
Author       : luoweiWHUT 1615108374@qq.com
Date         : 2023-11-09 22:16:58
LastEditors  : luoweiWHUT 1615108374@qq.com
LastEditTime : 2023-11-10 11:41:51
FilePath     : \EDA_competition\main.py
Description  : 批量运行demo,测试得分
'''
import os
from data_parse import Parser
cell_spi_path = "public/cells.spi"
paser = Parser()
paser.parse(cell_spi_path)
for cell_name in paser.cell_dict:
    os.system(f"python demo.py public/cells.spi {cell_name}")
