from datetime import datetime
import pll
import sys
import getopt
import numpy as np
from random import randint
import time

map_file_name = ""
order_mode = 0

def entrance(mapName):
    # 构建三个列表记录三种指标
    BFS_ary = []
    Index_ary = []
    query_ary = []
    # print(np.shape(BFS_ary))
    # 根据命令行输入取出图的名字
    map_file_name = mapName
    # print(f"mapName:{map_file_name}")

    # 根据不同的order构建Index=>记录指标=>查询=>记录指标

    # random暂时记为0
    BFS_ary.append(0)
    Index_ary.append(0)
    query_ary.append(0)
    for i in range(2,6):
        # build Index
        BFS_time,Average_Index_Size = build(map_file_name,i)
    # print(BFS_time,Average_Index_Size)
        BFS_ary.append(float("%.4f"%BFS_time))
        Index_ary.append(float("%.4f"%Average_Index_Size))

        # query
        query_time = query(map_file_name)
        query_ary.append(float("%.4f"%query_time))

    # 迭代一次2-hop
    # BFS_time,Average_Index_Size = build(map_file_name,5)
    # BFS_ary.append(float("%.4f"%BFS_time))
    # Index_ary.append(float("%.4f"%Average_Index_Size))
    # query_time = query(map_file_name)
    # query_ary.append(float("%.4f"%query_time))

    return BFS_ary,Index_ary,query_ary

# 使用pll中的build
def build(map_file_name,i):
    build_class = pll.PrunedLandmarkLabeling(map_file_name, i)
    return build_class.BFS_time, build_class.Average_Index_Size

    # pll.build(argv）

 # 初始化图信息
def query(map_file_name):
    query_class = pll.PrunedLandmarkLabeling()
    G = query_class.read_graph(map_file_name)
    nodes_list = list(G.nodes())
    nNodes = len(nodes_list)
    src_index = 0
    dest_index = 0

    # 进行10W次查询并记录时间
    start_time = time.time()
    for i in range(100000):
        src_index = randint(0,nNodes-1)
        dest_index = randint(0,nNodes-1)
        # print(f"src_index:{src_index}")
        query_class.query(nodes_list[src_index],nodes_list[dest_index])
    query_time = time.time()-start_time
    return query_time
