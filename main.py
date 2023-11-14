'''
Author       : luoweiWHUT 1615108374@qq.com
Date         : 2023-11-09 22:16:58
LastEditors  : luoweiWHUT 1615108374@qq.com
LastEditTime : 2023-11-14 21:04:02
FilePath     : \EDA_competition\main.py
Description  : 批量运行demo,测试得分
'''
import os
import time
import csv
from tqdm import tqdm
from data_parse import Parser
from public.evaluator import evaluator_case
cell_spi_path = "public/cells.spi"
save_dir = 'output'
paser = Parser()
paser.parse(cell_spi_path)
with open('result.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['cell name', 'score', 'width',
                     'bbox', 'pin_access', 'symmetric', 'drc', 'runtime'])
    time_cost = 0
    for cell_name in tqdm(paser.cell_dict):
        start = time.time()
        os.system(
            f"python TransistorPlacer.py {cell_spi_path} {cell_name} {save_dir}/{cell_name}.json")
        # os.system(
        #     f"TransistorPlacer.exe {cell_spi_path} {cell_name} {save_dir}/{cell_name}.json")
        run_time = time.time()-start
        time_cost += run_time
        score_list = evaluator_case(
            f'{save_dir}/{cell_name}.json', cell_name, cell_spi_path, run_time)
        writer.writerow([cell_name]+[score_list])
    print(f"求解总耗时:{time_cost}s")
