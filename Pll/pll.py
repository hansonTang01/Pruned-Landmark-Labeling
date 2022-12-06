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
import numpy as np
# import taichi as ti
import concurrent.futures
# ti.init(arch = ti.cpu)
index_file_path = "pll.idx"
max_length = 999999999


#Build调用： PrunedLandmarkLabeling(map_file_name, order_mode, False, is_multi_process)
# @ti.data_oriented
class PrunedLandmarkLabeling(object):
    '''
        attribute: 
        graph: 图    
        index    
        vertex_order： BFS节点顺序
        map_file_name
        build_order_time
    '''
    def __init__(self, map_file_name = ""):
        super(PrunedLandmarkLabeling, self).__init__()
        self.map_file_name = map_file_name
        self.graph = self.read_graph(map_file_name)
        self.index = {}
        self.Average_Index_Size = 0
        self.BFS_time = 0
        self.BFS_num_list = {}
        # self.nodes_list = []
        self.build_order_time = 0
    # 将构建好的index和order写入pll.idx，将index和order单独写入一个文件
    def write_index(self, map_file_name):

        # 写入pll.idx
        f = open(index_file_path, 'w')
        write_data = json.dumps(self.index)
        index_size = len(write_data)
        f.write(write_data)
        f.write('\n')
        f.write(json.dumps(self.vertex_order))
        f.close()

        # index写入单独文件
        fileName = "./idx_list/"+map_file_name+"_index.txt"
        f = open(fileName,"a")
        f.write(write_data)
        f.write("\n")
        f.close()

        # order写入单独文件
        fileName = "./idx_list/"+map_file_name+"_order.txt"
        f = open(fileName,"a")
        f.write(json.dumps(self.vertex_order))
        f.write("\n")
        f.close()
        self.Average_Index_Size= index_size/len(self.graph.nodes())
        print(f"Average Index Size: {(self.Average_Index_Size):.4f} Bytes ")
    
    # def write_2_hop_count(self,map_file_name):
    #     fileName = "./idx_list/"+ self.map_file_name+ "_2_hop_node_count.idx"
    #     f = open(fileName, 'w')
    #     write_data = json.dumps(nodes_list)
    #     f.write(write_data)
    #     f.close()
    # 将每轮BFS遍历的节点个数写入文件
    def write_BFS_num_list(self, map_file_name, BFS_num_list):
        fileName = "./idx_list/"+map_file_name+"_each_BFS_num.txt"
        f = open(fileName, 'a')
        write_data = json.dumps(BFS_num_list)
        f.write(write_data)
        f.write('\n')
        f.close()

    # 将图的节点和变数写入文件
    def write_map_info(self,map_file_name, map_info):
         fileName = "./idx_list/"+map_file_name+"_map_info.txt"
         f = open(fileName, 'w')
         write_data = json.dumps(map_info)
         f.write(write_data)
         f.write('\n')
         f.close()
        
    # 使用networkx读入图
    def read_graph(self, map_file_name):
        map_info = {}
        G = nx.DiGraph()
        f = open("map_file/" + map_file_name, 'r')
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
        # 输出节点和边的个数
        print(f"nodes:{len(G.nodes())}   edges:{len(G.edges())}")
        map_info["nodes"] = len(G.nodes())
        map_info['edges'] = len(G.edges())
        self.write_map_info(map_file_name,map_info)
        return G

    # 查询俩点间的距离
   
    def query(self, src, dest):
        #src_idx = self.index.get(src,None)
        #dest_idx = self.index.get(dest,None)
        #if src_idx and dest_idx:
        #    src_list = src_idx.get("backward",None)
        #    dest_list = dest_idx.get("forward",None)
        # print(self.index)
        
        src_list = self.index[src]["backward"]
        dest_list = self.index[dest]["forward"]
        # print(src,dest)
        # print(src_list)
        # print(dest_list)
        i = 0
        j = 0
        
        shortest_dist = max_length
        hop_nodes = []
        
        # 构建好的index里label是按照nodes_list里的节点顺序排序的，所以可以这么写
        while i < len(src_list) and j < len(dest_list):
            
            if src_list[i][0] == dest_list[j][0]:
                # print(src_list[i][0] == dest_list[j][0])
                curr_dist = src_list[i][1] + dest_list[j][1]
                if(curr_dist == 0 or curr_dist == 1):
                    hop_nodes.clear()
                    shortest_dist = curr_dist
                    break
                # print(curr_dist)
                # 当前距离未必为最小，若找到更小的距离，更新最小距离，之前的 hop_nodes作废，用新的代替
                if curr_dist < shortest_dist:
                    shortest_dist = curr_dist
                    hop_nodes.clear()
                    hop_nodes.append(src_list[i][0])   
                # 假定当前距离为最小，相等的点被暂时加入到 hop_nodes中
                elif curr_dist == shortest_dist:
                    hop_nodes.append(src_list[i][0])
                i += 1
                j += 1
        
            elif self.vertex_order[src_list[i][0]] > self.vertex_order[dest_list[j][0]]:
                    i += 1
            else:
                    j += 1
        # print(shortest_dist)
        # print(hop_nodes)
        return shortest_dist, hop_nodes

    # @ti.func
    def gen_2_hop_list(self,src,dest):
        # print(type(src))
        # print(self.index)
        # print(self.index[src])
        src_list = self.index[src]["backward"]
        dest_list = self.index[dest]["forward"]
        return src_list,dest_list

    # @ti.func
    def query_for_2_hop(self, src):
        #src_idx = self.index.get(src,None)
        #dest_idx = self.index.get(dest,None)
        #if src_idx and dest_idx:
        #    src_list = src_idx.get("backward",None)
        #    dest_list = dest_idx.get("forward",None)
        # print(self.index)
        count_result = {}
        nodes_list = list(self.graph.nodes())
        for node in nodes_list:
            count_result[node] = 0
        src_list = self.index[src]["backward"]
        for dest in nodes_list:
            i = 0
            j = 0
            shortest_dist = max_length
            hop_nodes = []
            dest_list = self.index[dest]["forward"]
            while i < len(src_list) and j < len(dest_list):
                if src_list[i][0] == dest_list[j][0]:
                    # print(src_list[i][0] == dest_list[j][0])
                    curr_dist = src_list[i][1] + dest_list[j][1]
                    if(curr_dist == 0 or curr_dist == 1):
                        hop_nodes.clear()
                        shortest_dist = curr_dist
                        break
                    # print(curr_dist)
                    # 当前距离未必为最小，若找到更小的距离，更新最小距离，之前的 hop_nodes作废，用新的代替
                    if curr_dist < shortest_dist:
                        shortest_dist = curr_dist
                        hop_nodes.clear()
                        hop_nodes.append(src_list[i][0])   
                    # 假定当前距离为最小，相等的点被暂时加入到 hop_nodes中
                    elif curr_dist == shortest_dist:
                        hop_nodes.append(src_list[i][0])
                    i += 1
                    j += 1
            
                elif self.vertex_order[src_list[i][0]] > self.vertex_order[dest_list[j][0]]:
                        i += 1
                else:
                        j += 1
            for hop_node in hop_nodes:
                count_result[hop_node]+=1   
            # print(count_result)
        return count_result
                # print(self.count_result)
        # print(shortest_dist)
        # print(hop_nodes)

    # 加载图的Index
    def load_index(self, index_file_path):
        f = open(index_file_path, 'r')
        data = f.readlines()
        # print(f"data:{data}")
        result = json.loads(data[0])
        # print(f"result:{result}")
        self.vertex_order = json.loads(data[1])
        self.index = result
        f.close()
        return result
        
    def gen_test_order(self):
        nodes_list = list(self.graph.nodes())
        # print(f"nodes:{self.graph.nodes}")
        # print(f"nodes_num:{nNodes}")
        result = self.generate_order_for_BFS(nodes_list)
        self.vertex_order = result

    def gen_random_order(self):
        # nodes_list为节点的数组
        nodes_list = list(self.graph.nodes())
        # print(nodes_list)
        random.shuffle(nodes_list)
        # print(nodes_list)
        result = self.generate_order_for_BFS(nodes_list)
        self.vertex_order = result

    def gen_degree_base_order(self):
        nodes_list = list(sorted(self.graph.degree, key=lambda x: x[1], reverse=True))
        # print(f"degree:{list(dict(nodes_list).values())}\n")

        result = self.generate_order_for_BFS(nodes_list)
        self.vertex_order = result

    def in_out_degree(self):
        nodes_list = list(self.graph.nodes())
        sort_list = []
        nNodes = len(nodes_list)
        in_degree = np.array(list(dict(self.graph.in_degree).values()))
        out_degree = np.array(list(dict(self.graph.out_degree).values()))
        in_out_degree = np.multiply(in_degree,out_degree)

        # print(self.graph.degree)
        # print(in_degree)
        # print(out_degree)
        for i in range(nNodes):
            sort_list.append((nodes_list[i],in_out_degree[i]))
        after_sort_list = list(sorted(sort_list, key=lambda x: x[1], reverse=True))
        # print(after_sort_list)
        result = self.generate_order_for_BFS(after_sort_list)
        self.vertex_order = result

    def gen_closeness_base_order(self):
        nodes_list = nx.closeness_centrality(self.graph)
        # print(f"closeness:{nodes_list}")
        nodes_list = list(sorted(nodes_list.items(), key=lambda item:item[1], reverse = True))
        # print(f"nodes_list:{nodes_list}")
        result = self.generate_order_for_BFS(nodes_list)
        self.vertex_order = result

    def gen_betweeness_base_order(self):
        nodes_list = nx.betweenness_centrality(self.graph, weight="weight")
        # print(f"betweenness:{nodes_list}")
        nodes_list = list(sorted(nodes_list.items(), key=lambda item:item[1], reverse = True))
        # print(f"nodes_list:{nodes_list}")
        result = self.generate_order_for_BFS(nodes_list)
        self.vertex_order = result

    '''
        Function: gen_2_hop_base_order
        description: 按照我们定义的算法找出2_hop点重新进行排序
    '''

    # def fetch_nodes_list(self):
    #     nodes_list = list(self.graph.nodes())
    #     return nodes_list()
    # def fetch_2_hop_count(self):
    #     return 
   
    
    '''
        src,dest: example: 24 24
    '''
    def cal_2_hop_count(self, nodes_list):
        nNodes = len(nodes_list)
        count = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(self.query_for_2_hop, nodes_list, chunksize= 128)
        print("************")
        for sub_list in result:
            for key,value in sub_list.items():
                self.count_result[key] += value
        # print(self.count_result)
        # print(result)
        # for src in nodes_list:
        #     self.query_for_2_hop(src)
        #     count += 1
        #     self.hop_base_process_bar(count,nNodes)
     
    # @ti.kernel
    '''
        nodes_list: example: [24, 1, 595, 1143, 1392, 1405, 156]
    '''
    def gen_4_dim_list(self):
        print(self.index)
        
    def gen_2_hop_base_order(self):
        # 这里为什么没有short_dist不会被access呢？
        # short_dist = 0
        # self.gen_4_dim_list()
        self.count_result = {}
        
        nodes_list = (list(self.graph.nodes()))
        # print(nodes_list)
        # print(f"nodes_list:{nodes_list}")
        # print(nodes_list)
        for node in nodes_list:
            self.count_result[node] = 0
        #  遍历节点，得到2跳列表
        # nodes_list = list(map(int,nodes_list))
        self.cal_2_hop_count(nodes_list)
        
        # print(f"count_result:{count_result}")
        # print("test!test!")
        # 排序
        nodes_list = list(sorted(self.count_result.items(), key=lambda item:item[1], reverse = True))
        # print(f"nodes_list:{nodes_list}")
        # 将count_result写入文件
        fileName = "./idx_list/"+ self.map_file_name+ "_2_hop_node_count.txt"
        f = open(fileName, 'w')
        write_data = json.dumps(nodes_list)
        f.write(write_data)
        f.close()

        self.vertex_order = self.generate_order_for_BFS(nodes_list)
    

    # 基于label计数
    def gen_label_count_base_order(self, mode = 0):
        count_result = {}
        nodes_list = list(self.graph.nodes())
        # print(f"nodes_list:{nodes_list}")
        for node in nodes_list:
            count_result[node] = 0

        temp_index = self.index
        for node in nodes_list:
            bd_list = temp_index[node]["backward"]
            fw_list = temp_index[node]["forward"]
            for i in range(len(bd_list)):
                count_result[bd_list[i][0]]+=1
            for i in range(len(fw_list)):
                count_result[fw_list[i][0]]+=1
            
        nodes_list = list(sorted(count_result.items(), key=lambda item:item[1], reverse = True))
        # print(f"nodes_list:{nodes_list}")
        # 将count_result写入文件
        fileName = "./idx_list/"+ self.map_file_name+ "_label_base_count.txt"
        f = open(fileName, 'a')
        write_data = json.dumps(nodes_list)
        f.write(write_data)
        f.write('\n')
        f.close()
        
        result = self.generate_order_for_BFS(nodes_list)
        self.vertex_order = result

    def hop_base_process_bar(self,count,nNodes):
        if (count == int(0.1*nNodes)):
            print("**********finish 10% of building 2-hop order*********\n")
        if (count == int(0.2*nNodes)):
            print("**********finish 20% of building 2-hop order*********\n")
        if (count == int(0.3*nNodes)):
            print("**********finish 30% of building 2-hop order*********\n")
        if (count == int(0.4*nNodes)):
            print("**********finish 40% of building 2-hop order*********\n")
        if (count == int(0.5*nNodes)):
            print("**********finish 50% of building 2-hop order*********\n")
        if (count == int(0.6*nNodes)):
            print("**********finish 60% of building 2-hop order*********\n")
        if (count == int(0.7*nNodes)):
            print("**********finish 70% of building 2-hop order*********\n")
        if (count == int(0.8*nNodes)):
            print("**********finish 80% of building 2-hop order*********\n")
        if (count == int(0.9*nNodes)):
            print("**********finish 90% of building 2-hop order*********\n")
        if (count == nNodes):
            print("**********finish 100% of building 2-hop order*********\n")
    # def feedback_tuning(self, mode = 0, w, b, ):
    #     BFS_traverse = self.fetch_order()
        
    # # 拿到上一次BFS遍历过程中的扩散
    # def fetch_order(self, mode = 1):
    #     result = {}
    #     nNodes = len(self.graph.nodes())
    #     fileName = "./idx_list"+ self.map_file_name + "_each_BFS_num.idx"
    #     f = open(fileName, 'r')
    #     raw_data = f.readlines()
    #     # 加载基于degree的BFS_count
    #     BFS_traverse = eval(raw_data[-1])
    #     return BFS_traverse
    #     # nodes_list = list(raw_data.keys())
    #     # for idx, v in enumerate(nodes_list):
    #     #     # print(f"idx:{idx}, v:{v}")
    #     #     result[v[0]] = nNodes - idx
    # # 生成节点的order

    '''
        生成了self.build_order_time, self.vertex_order
    '''
    def gen_order(self, mode = 0):
        # 根据输入的mode采取不同的策略构建节点的order
        # mode = 1 => sequential order（图的输入顺序）
        # mode = 2 => 随机  
        # mode = 3 => 基于degree  
        # mode = 4 => 基于betweenness     
        # mode = 5 => 基于2-hop
        start_time_order = time.time()
        if (mode == 0):
            self.gen_test_order()
        if (mode == 1):
            print("\n*************Random****************")
            self.gen_random_order()
        if (mode == 2):
            print("\n*************Degree****************")
            self.gen_degree_base_order()
        if (mode == 3):
            print("\n*************Clossness*************")
            self.gen_closeness_base_order()
        if (mode == 4):
            print("\n*************Betweenness***********")
            self.gen_betweeness_base_order()
        if (mode == 5):
            print("\n*************2-hop-based***********")
            self.gen_2_hop_base_order()
        if (mode == 6):
            print("\n********label-count-based**********")
            self.gen_label_count_base_order()
        if (mode == 7):
            print("\n*********in-out-degree*************")
            self.in_out_degree()
        # self.vertex_order = {k: v for k, v in sorted(self.vertex_order.items(), key=lambda item: -item[1])}
        self.build_order_time = time.time()-start_time_order
        print("finish generating order")
        print(f"Time cost : {self.build_order_time:.4f}")
        print("***********************************")
        
        # print(self.vertex_order)
        # print("")

    def generate_order_for_BFS(self,nodes_list):
        result = {}
        nNodes = len(self.graph.nodes())
        for idx, v in enumerate(nodes_list):
            result[v[0]] = nNodes - idx
        return result
        
    # 判断是否需要剪枝
    # @ti.func
    def need_to_expand(self, src, dest, dist = -1):
        # print("nx: %s -> %s: %d" % (src, dest, v))
        # print(src,dest,dist)
        our_result,_ = self.query(src, dest)
       

        # print(our_result)
        v = dist
        # print("pll: %s -> %s: %d" % (src, dest, our_result))
        if (our_result <= v):
            # print(0)
            return False
        return True

    # 进行BFS
    '''
        生成了self.index, self.BFS_num_list, self.BFS_time
        无返回值
    '''
    def build_index(self):
        # print(self.vertex_order)
        # 构建BFS_num记录每轮BFS遍历的节点个数
        vertex_order = self.vertex_order
        nodes_list = list(vertex_order.keys())
        
        # for index, node in enumerate(self.vertex_order):
        #     self.BFS_num_list[node] = 0
        for i in range(len(nodes_list)):
            self.BFS_num_list[nodes_list[i]] = 0
        # print(self.BFS_num_list)
        self.index = {}
        has_process = {}
        pq = Q.PriorityQueue()
        # print(f'pq:{pq}\n')
        for v in sorted(list(self.graph.nodes())):
            # backward-0   forward-1
            self.index[v] = {"backward": [], "forward": []}
            has_process[v] = False

        # print(self.index)
        # print(f"self.index:{self.index}")
        # print(f"has_process:{has_process}")
        i = 0
        nNode = len(self.graph.nodes())
        count = 0

         # 利用Order开始BFS并记录时间
        start_time_BFS = time.time()
        for order_item in self.vertex_order.items():
            cur_node = order_item[0]
            # print(f"cur_NODE:{cur_node}")
            i += 1
            # Calculate Forward
            if (i%1000 == 0) :
                print("Caculating %s (%d/%d) forward ... " % (cur_node, i, nNode))
            pq.put((0, cur_node))
            # 把所有点是否剪枝记为0
            for k in has_process:
                has_process[k] = False
                # print(f"k:{k}")
            # print(f"has_process:{has_process}")
            while (not pq.empty()):
                cur_dist, src = pq.get()

                # print("Pop: (%s %d)"%(src,cur_dist))
                # print(has_process[src])
                # print(self.vertex_order[cur_node] < self.vertex_order[src])
                # print(not self.need_to_expand(cur_node, src, cur_dist))
                # print(cur_node,src,cur_dist)
                if (has_process[src] or self.vertex_order[cur_node] < self.vertex_order[src] or not self.need_to_expand(cur_node, src, cur_dist)):
                    # print(f"self.vertex_order[cur_node]:{self.vertex_order[cur_node]}")
                    # print(f'src:{src}')
                    
                    has_process[src] = True
                    continue
                count+=1
                # print(count)
                has_process[src] = True
                self.index[src]["forward"].append((cur_node, cur_dist))
                # print(f"index:{self.index}")
                edges = self.graph.out_edges(src)
                # print(src)
                # print(f"edges:{edges}")
                for _, dest in edges:
                    # print(f"dest: {dest}")
                    weight = self.graph.get_edge_data(src, dest)['weight']
                    if (has_process[dest]):
                        continue
                    pq.put((cur_dist + weight, dest))
                    # print("Push: (%s, %d)"%(dest, cur_dist + weight))

            # Calculate Backward
            if (i%1000 == 0) :
                print("Caculating %s (%d/%d) backward..." % (cur_node, i, nNode))
            pq.put((0, cur_node))
            for k in has_process:
                has_process[k] = False
            while (not pq.empty()):
                cur_dist, src = pq.get()
                # print("Pop: (%s %d)"%(src,cur_dist))
                if (has_process[src] or self.vertex_order[cur_node] < self.vertex_order[src] or not self.need_to_expand(src, cur_node, cur_dist)):
                    continue
                count+=1
                has_process[src] = True
                self.index[src]["backward"].append((cur_node, cur_dist))
                edges = self.graph.in_edges(src)
                # print(src)
                # print(edges)
                for dest, _ in edges:
                    weight = self.graph.get_edge_data(dest, src)['weight']
                    if (has_process[dest]):
                        continue
                    pq.put((cur_dist + weight, dest))
                    # print("Push: (%s, %d)"%(dest, cur_dist + weight))
            self.BFS_num_list[cur_node] = count
            # print(self.BFS_num_list)
            count = 0
        # 记录BFS时间
        # print(f"index:{self.index}")
        # print(f"order:{self.vertex_order}")
        print(f'finish building index')
        self.BFS_time = time.time() - start_time_BFS
        print(f'Time cost: {(self.BFS_time):.4f}')
        print("***********************************")

    '''
        Function: feedback
        order: e.g ['a','b','c','d']
        BFS_raverse_record: 代表
    '''
    def feedback(self, order, w, k_feedback, b):
        # 构建BFS_num记录每轮BFS遍历的节点个数
        print(f"*********\nw:{w}\n*******")
        print(f"*********\nk_feedback:{k_feedback}\n*******")
        print(f"*********\nb:{b}\n*******")
        self.vertex_order = self.generate_order_for_BFS(order)
        self.BFS_num_list = {}
        BFS_traverse_record = []
        changeSet = []
        for index, node in enumerate(self.vertex_order):
            self.BFS_num_list[node] = 0 
        has_process = {}
        pq = Q.PriorityQueue()
        # print(f'pq:{pq}\n')
        for v in self.graph.nodes():
            self.index[v] = {"backward": [], "forward": []}
            has_process[v] = False
        # print(f"self.index:{self.index}")
        # print(f"has_process:{has_process}")
        i = -1
        nNode = len(self.graph.nodes())
        count = 0

         # 利用Order开始BFS并记录时间
        start_time_BFS = time.time()
        for order_item in self.vertex_order.items():
            cur_node = order_item[0]
            # print(f"cur_NODE:{cur_node}")
            i += 1
            if(i== k_feedback+w):
                break
            # Calculate Forward
            if (i%1000 == 0) :
                print("Caculating %s (%d/%d) forward ... " % (cur_node, i, nNode))
            pq.put((0, cur_node))
            # 把所有点是否剪枝记为0
            for k in has_process:
                has_process[k] = False
                # print(f"k:{k}")
            # print(f"has_process:{has_process}")
            while (not pq.empty()):
                cur_dist, src = pq.get()
                # print("Pop: (%s %d)"%(src,cur_dist))
                if (has_process[src] or self.vertex_order[cur_node] < self.vertex_order[src] or not self.need_to_expand(cur_node, src, cur_dist)):
                    # print(f"self.vertex_order[cur_node]:{self.vertex_order[cur_node]}")
                    # print(f'src:{src}')
                    
                    has_process[src] = True
                    continue
                count+=1
                has_process[src] = True
                # print(f"index:{self.index}")
                edges = self.graph.out_edges(src)
                # print(src)
                # print(f"edges:{edges}")
                for _, dest in edges:
                    # print(f"dest: {dest}")
                    weight = self.graph.get_edge_data(src, dest)['weight']
                    if (has_process[dest]):
                        continue
                    pq.put((cur_dist + weight, dest))
                    # print("Push: (%s, %d)"%(dest, cur_dist + weight))

            # Calculate Backward
            if (i%1000 == 0) :
                print("Caculating %s (%d/%d) backward..." % (cur_node, i, nNode))
            pq.put((0, cur_node))
            for k in has_process:
                has_process[k] = False
            while (not pq.empty()):
                cur_dist, src = pq.get()
                # print("Pop: (%s %d)"%(src,cur_dist))
                if (has_process[src] or self.vertex_order[cur_node] < self.vertex_order[src] or not self.need_to_expand(src, cur_node, cur_dist)):
                    continue
                count+=1
                has_process[src] = True
                edges = self.graph.in_edges(src)
                # print(src)
                # print(edges)
                for dest, _ in edges:
                    weight = self.graph.get_edge_data(dest, src)['weight']
                    if (has_process[dest]):
                        continue
                    pq.put((cur_dist + weight, dest))
                    # print("Push: (%s, %d)"%(dest, cur_dist + weight))
            self.BFS_num_list[cur_node] = count
            BFS_traverse_record.append(count)
            if (i >= 2*w and i <= k_feedback+w-1):
                if self.is_need_change_order(i-w, BFS_traverse_record, w, b):
                    changeSet.append(i-w)
            count = 0
        # 记录BFS时间
        print(f'finish one time feedback_tuning')
        self.BFS_time = time.time() - start_time_BFS
        print(f'Time cost: {(self.BFS_time):.4f}')
        print("***********************************")
        return BFS_traverse_record,changeSet
        
    def is_need_change_order(self, cur,BFS_traverse_record, w, b):
        if ((sum(BFS_traverse_record[cur-w:cur])/w) / BFS_traverse_record[cur]> (1+b)) and ((sum(BFS_traverse_record[cur+1:cur+w+1])/w)/BFS_traverse_record[cur]>(1+b)):
            # if(BFS_traverse_record[cur]<100):
                # print(BFS_traverse_record[cur-w:cur])
                # print(BFS_traverse_record[cur])
            return True
        return False




    

