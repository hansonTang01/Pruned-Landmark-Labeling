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
    build_order_time_list = []
    # print(np.shape(BFS_ary))
    # 根据命令行输入取出图的名字
    map_file_name = mapName
    pll_class = pll.PrunedLandmarkLabeling(map_file_name)
    # print(f"mapName:{map_file_name}")

    # 根据不同的order构建Index=>记录指标=>查询=>记录指标

    # random暂时记为0
    # BFS_ary.append(0)
    # Index_ary.append(0)
    # query_ary.append(0)

   
    # 基于degree
    BFS_time,Average_Index_Size,build_order_time = build(map_file_name,2,pll_class)
    BFS_ary.append(float("%.4f"%BFS_time))
    Index_ary.append(float("%.4f"%Average_Index_Size))
    build_order_time_list.append(float("%.4f"%build_order_time))
    # query
    query_time = query(map_file_name,pll_class)
    query_ary.append(float("%.4f"%query_time))

    # 基于degree-base-label-count
    BFS_time,Average_Index_Size,build_order_time = build(map_file_name,6,pll_class)
    BFS_ary.append(float("%.4f"%BFS_time))
    Index_ary.append(float("%.4f"%Average_Index_Size))
    build_order_time_list.append(float("%.4f"%build_order_time))
    # query
    query_time = query(map_file_name,pll_class)
    query_ary.append(float("%.4f"%query_time))

    # 基于in-out-degree
    BFS_time,Average_Index_Size,build_order_time = build(map_file_name,7,pll_class)
    BFS_ary.append(float("%.4f"%BFS_time))
    Index_ary.append(float("%.4f"%Average_Index_Size))
    build_order_time_list.append(float("%.4f"%build_order_time))
    # query
    query_time = query(map_file_name,pll_class)
    query_ary.append(float("%.4f"%query_time))

    # 基于in-outdegree-base-label-count
    BFS_time,Average_Index_Size,build_order_time = build(map_file_name,6,pll_class)
    BFS_ary.append(float("%.4f"%BFS_time))
    Index_ary.append(float("%.4f"%Average_Index_Size))
    build_order_time_list.append(float("%.4f"%build_order_time))
    # query
    query_time = query(map_file_name,pll_class)
    query_ary.append(float("%.4f"%query_time))

    # 基于betweenness
    BFS_time,Average_Index_Size,build_order_time = build(map_file_name,4,pll_class)
    BFS_ary.append(float("%.4f"%BFS_time))
    Index_ary.append(float("%.4f"%Average_Index_Size))
    build_order_time_list.append(float("%.4f"%build_order_time))
    # query
    query_time = query(map_file_name,pll_class)
    query_ary.append(float("%.4f"%query_time))

    # 基于betweenness-2-hop
    BFS_time,Average_Index_Size,build_order_time = build(map_file_name,5,pll_class)
    BFS_ary.append(float("%.4f"%BFS_time))
    Index_ary.append(float("%.4f"%Average_Index_Size))
    build_order_time_list.append(float("%.4f"%build_order_time))
    # query
    query_time = query(map_file_name,pll_class)
    query_ary.append(float("%.4f"%query_time))

    # # 基于betweenness-base-label-count
    # BFS_time,Average_Index_Size,build_order_time = build(map_file_name,6,pll_class)
    # BFS_ary.append(float("%.4f"%BFS_time))
    # Index_ary.append(float("%.4f"%Average_Index_Size))
    # build_order_time_list.append(float("%.4f"%build_order_time))
    # # query
    # query_time = query(map_file_name,pll_class)
    # query_ary.append(float("%.4f"%query_time))

    # 反馈调节

    # BFS_time,Average_Index_Size = feedback_tuning(map_file_name,5)
    # BFS_ary.append(float("%.4f"%BFS_time))
    # Index_ary.append(float("%.4f"%Average_Index_Size))
    # query_time = query(map_file_name,pll_class)
    # query_ary.append(float("%.4f"%query_time))
   
    return BFS_ary,Index_ary,query_ary,build_order_time_list
    # 迭代一次2-hop
    # BFS_time,Average_Index_Size = build(map_file_name,5)
    # BFS_ary.append(float("%.4f"%BFS_time))
    # Index_ary.append(float("%.4f"%Average_Index_Size))
    # query_time = query(map_file_name,pll_class)
    # query_ary.append(float("%.4f"%query_time))

# 使用pll中的build
def build(map_file_name,i,pll_class):
    pll_class.gen_order(i)
    pll_class.build_index()
    pll_class.write_index(map_file_name)
    pll_class.write_BFS_num_list(map_file_name, pll_class.BFS_num_list)
    return pll_class.BFS_time, pll_class.Average_Index_Size,pll_class.build_order_time

    # pll.build(argv）

def build_for_2_hop_order(map_file_name):
    hop_order_class = pll.PrunedLandmarkLabeling(map_file_name)
    hop_order_class.gen_order(5)
    hop_order_class.fetch_nodes_list()
    hop_order_class.load_index('pll.py')
    hop_order_class.build_index()
#基于扩散数下降进行调节
def feedback_tuning(map_file_name,  w):
    print("\n********feedback_tuning************")
    feedback_class = pll.PrunedLandmarkLabeling(map_file_name)
    nNodes = len(feedback_class.graph.nodes())
    k = int(nNodes*0.2)
    order = list(sorted(feedback_class.graph.degree, key=lambda x: x[1], reverse=True))
    # print(f"initial order: {order}")
    changeSet = [1,2]  
    for i in range(40):
        BFS_traverse_record, changeSet = feedback_class.feedback(order,w, k, 0.3)
        print(f"****************\n{BFS_traverse_record}\n***********")
        print(f"****************\n{changeSet}\n***********")
        
        for j in range(len(changeSet)):
            pop_node = order.pop(changeSet[j])
            order.insert(k, pop_node)
    # print(f"order:{order}")
    vertex_order = generate_order_for_BFS(order)
    feedback_class.vertex_order = vertex_order
    feedback_class.build_index()
    feedback_class.write_index(map_file_name)
    feedback_class.write_BFS_num_list(map_file_name, feedback_class.BFS_num_list)
    return feedback_class.BFS_time, feedback_class.Average_Index_Size,feedback_class.build_order_time


 # 查询
def query(map_file_name,pll_class):

    nodes_list = list(pll_class.graph.nodes())
    # print(nodes_list)
    # print(pll_class.vertex_order)
    nNodes = len(nodes_list)
    src_index = 0
    dest_index = 0

    # 进行10W次查询并记录时间
    start_time = time.time()
    for i in range(100000):
        src_index = randint(0,nNodes-1)
        dest_index = randint(0,nNodes-1)
        # print(f"src_index:{src_index}")
        # print(src_index,dest_index)
        # print(nodes_list[src_index],nodes_list[dest_index],nodes_list[0])
        pll_class.query(nodes_list[src_index],nodes_list[dest_index])
    query_time = time.time()-start_time
    return query_time

def generate_order_for_BFS(nodes_list):
    result = {}
    nNodes = len(nodes_list)
    for idx, v in enumerate(nodes_list):
        result[v[0]] = nNodes - idx
    return result

if __name__== "__main__":
    entrance("macau.map")