'''
Author       : luoweiWHUT 1615108374@qq.com
Date         : 2023-11-09 22:16:58
LastEditors  : luoweiWHUT 1615108374@qq.com
LastEditTime : 2023-11-10 20:01:37
FilePath     : \EDA_competition\main.py
Description  : 批量运行demo,测试得分
'''
import os
import time
from tqdm import tqdm
from data_parse import Parser
cell_spi_path = "public/cells.spi"
paser = Parser()
paser.parse(cell_spi_path)
start = time.time()
for cell_name in tqdm(paser.cell_dict):
    os.system(f"python demo.py public/cells.spi {cell_name}")
    # os.system(
    #     f"TransistorPlacer.exe public/cells.spi {cell_name} output/{cell_name}.json")
print(f"耗时:{time.time()-start}s")
