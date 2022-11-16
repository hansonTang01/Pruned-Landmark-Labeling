import pll
import sys
import time
import os
import networkx as nx
import random
#import pylab
import queue as Q
import json
import threading
import multiprocessing
import getopt
from random import randint

help_msg = "python pll.py query -i map"

try:
    options, args = getopt.getopt(sys.argv[1:], "hi:", ["help","map="])
    for name, value in options:
        if name in ("-h", "--help"):
            print(help_msg)
            exit()
        if name in ("-i", "--input"):
            map_file_name = value
except:
    print(help_msg)
    exit()

G = nx.DiGraph()
f = open(map_file_name, 'r')
data = f.readlines()
f.close()
for idx, lines in enumerate(data):
    # 这里的2，看map后缀的文件就可以明白， 数据是从第三行开始的
    if (idx < 2):
        continue
    # （src, dest, dist, is_one_way）为osmnx上下载的图的格式
    src, dest, dist, is_one_way = lines.split(" ")
    G.add_weighted_edges_from([(src, dest, int(dist))])
    #  is_one_way=0 => 为无向图
    if (int(is_one_way) == 0):
        G.add_weighted_edges_from([(dest, src, int(dist))])

nodes_list = list(G.nodes())
nNodes = len(nodes_list)
src_index = 0
dest_index = 0
test = pll.PrunedLandmarkLabeling()
start_time = time.time()

# 十万次查询的时间
for i in range(100000):
    src_index = randint(0,nNodes-1)
    dest_index = randint(0,nNodes-1)
    # print(f"src_index:{src_index}")
    test.query(nodes_list[src_index],nodes_list[dest_index])

# print(ppl.query(src_vertex, target_vertex))

print("Total time: %f" % (time.time() - start_time))
exit()

