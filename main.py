'''
Author       : luoweiWHUT 1615108374@qq.com
Date         : 2023-11-09 22:16:58
LastEditors  : luoweiWHUT 1615108374@qq.com
LastEditTime : 2023-11-15 17:27:54
FilePath     : \EDA_competition\main.py
Description  : 批量运行demo,测试得分
'''
import os
import time
import csv
import sys
from tqdm import tqdm
from data_parse import Parser
from my_evaluator import evaluator_case
if __name__ == "__main__":
    cell_spi_path = "public/cells.spi"
    if len(sys.argv) < 2:
        save_dir = 'output'
        outcsv_path = 'result.csv'
    else:
        save_dir = 'output_cplus'
        outcsv_path = 'result_cplus.csv'
    paser = Parser()
    paser.parse(cell_spi_path)
    with open(outcsv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['cell name', 'score', 'width',
                         'bbox', 'pin_access', 'symmetric', 'drc', 'runtime'])
    time_cost = 0
    for i, cell_name in enumerate(tqdm(paser.cell_dict)):
        start = time.time()
        if len(sys.argv) < 2:
            os.system(
                f"python TransistorPlacer.py {cell_spi_path} {cell_name} {save_dir}/{cell_name}.json")
        else:
            os.system(
                f"TransistorPlacer.exe {cell_spi_path} {cell_name} {save_dir}/{cell_name}.json")
        run_time = time.time()-start
        time_cost += run_time
        score_list = evaluator_case(
            f'{save_dir}/{cell_name}.json', cell_name, cell_spi_path, run_time)
        if score_list:
            with open(outcsv_path, 'a') as f:
                writer = csv.writer(f)
                score_list.insert(0, cell_name)
                writer.writerow(score_list)
        else:
            print(f"{cell_name} run error!")
    print(f"求解总耗时:{time_cost}s")
