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
    # BFS_ary.append(0)
    # Index_ary.append(0)
    # query_ary.append(0)

    # for i in range(2,7):
    #     # build Index
    #     BFS_time,Average_Index_Size = build(map_file_name,i)
    # # print(BFS_time,Average_Index_Size)
    #     BFS_ary.append(float("%.4f"%BFS_time))
    #     Index_ary.append(float("%.4f"%Average_Index_Size))

    #     # query
    #     query_time = query(map_file_name)
    #     query_ary.append(float("%.4f"%query_time))

    BFS_time,Average_Index_Size = feedback_tuning(map_file_name,5)
    # BFS_ary.append(float("%.4f"%BFS_time))
    # query_time = query(map_file_name)
    # query_ary.append(float("%.4f"%query_time))
    
    return BFS_ary,Index_ary,query_ary
    # 迭代一次2-hop
    # BFS_time,Average_Index_Size = build(map_file_name,5)
    # BFS_ary.append(float("%.4f"%BFS_time))
    # Index_ary.append(float("%.4f"%Average_Index_Size))
    # query_time = query(map_file_name)
    # query_ary.append(float("%.4f"%query_time))

# 使用pll中的build
def build(map_file_name,i):
    build_class = pll.PrunedLandmarkLabeling(map_file_name)
    build_class.gen_order(i)
    build_class.build_index()
    build_class.write_index(map_file_name)
    build_class.write_BFS_num_list(map_file_name, build_class.BFS_num_list)
    return build_class.BFS_time, build_class.Average_Index_Size

    # pll.build(argv）

#基于扩散数下降进行调节
def feedback_tuning(map_file_name,  w):
    feedback_class = pll.PrunedLandmarkLabeling(map_file_name)
    nNodes = len(feedback_class.graph.nodes())
    k = int(nNodes*0.1)
    order = list(sorted(feedback_class.graph.degree, key=lambda x: x[1], reverse=True))
    changeSet = [1,2]  
    for i in range(10):
        while(changeSet!=[]):
            BFS_traverse_record, changeSet = feedback_class.feedback(order,w, k, 0.3)
            for j in range(len(changeSet)):
                order.insert(k,order.pop(changeSet[j]))
    vertex_order = generate_order_for_BFS(order)
    feedback_class.vertex_order = vertex_order
    feedback_class.build_index()
    feedback_class.write_index(map_file_name)
    feedback_class.write_BFS_num_list(map_file_name, feedback_class.BFS_num_list)
    return feedback_class.BFS_time, feedback_class.Average_Index_Size


 # 查询
def query(map_file_name):
    query_class = pll.PrunedLandmarkLabeling(map_file_name)
    query_class.load_index("pll.idx")
    nodes_list = list(query_class.graph.nodes())
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

def generate_order_for_BFS(nodes_list):
    result = {}
    nNodes = len(nodes_list)
    for idx, v in enumerate(nodes_list):
        result[v[0]] = nNodes - idx
    return result

if __name__== "__main__":
    entrance("test.map")