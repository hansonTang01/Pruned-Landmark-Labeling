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

index_file_path = "pll.idx"
max_length = 999999999

#Build调用： PrunedLandmarkLabeling(map_file_name, order_mode, False, is_multi_process)
class PrunedLandmarkLabeling(object):
    '''
        attribute: 
        graph: 图    
        index    
        vertex_order： BFS节点顺序

    '''
    def __init__(self, map_file_name = "", order_mode = 0, validation = False, is_multi_process = False):
        super(PrunedLandmarkLabeling, self).__init__()
        if (not validation):
            # build——构建Index
            if (map_file_name != ""):
                # 读取图并计算读取的时间
                start_time_readGraph = time.time()
                self.map_file_name = map_file_name
                self.graph = self.read_graph(map_file_name)
                # print(f"finish Reading graph")
                # print(f"Time cost of Reading graph is {time.time()-start_time_readGraph}")
                # print("***********************************")
                # 构建Index
                if (is_multi_process):
                    # 使用多进程——这部分代码还没看
                    self.index = self.build_index_multi_process(order_mode)
                else:
                    # 不使用多进程
                    self.index, start_time_BFS = self.build_index(order_mode,map_file_name)
                    print(f'finish building index')
                    self.BFS_time = time.time() - start_time_BFS
                    print(f'Time cost: {(self.BFS_time):.4f}')
                    print("***********************************")
                    # print(f"Index Size: {self.index_size} Bytes ")
                    self.Average_Index_Size= self.index_size/len(self.graph.nodes())
                    print(f"Average Index Size: {(self.Average_Index_Size):.4f} Bytes ")

            # Query——查询俩点间的距离
            else:
                self.index = self.load_index(index_file_path)
        # validation——验证(这部分代码还没看)
        else:
            self.graph = self.read_graph(map_file_name)
            self.index = self.load_index(index_file_path)

    # 将构建好的index和order写入pll.idx
    def write_index(self, map_file_name):
        f = open(index_file_path, 'w')
        # f.writelines(str(len(self.graph.nodes)) + "\n")
        # print("Index:")
        # for k in self.index:
        #     print(k)
        #     print(self.index[k])
        write_data = json.dumps(self.index)
        self.index_size = len(write_data)
        f.write(write_data)
        f.write('\n')
        f.write(json.dumps(self.vertex_order))
        f.close()
        fileName = "./idx_list/"+map_file_name+"_index.idx"
        f = open(fileName,"a")
        f.write(write_data)
        f.write("\n")
        f.close()
    
    # 将每轮BFS遍历的节点个数写入文件
    def write_BFS_num_list(self, map_file_name, BFS_num_list):
        fileName = "./idx_list/"+map_file_name+"_each_BFS_num.idx"
        f = open(fileName, 'a')
        write_data = json.dumps(BFS_num_list)
        f.write(write_data)
        f.write('\n')
        f.close()

    # 使用networkx读入图
    def read_graph(self, map_file_name):
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

        # print("G.edges:")
        # print(G.edges())
        # print("G.nodes:")
        # print(G.nodes())
        # print("")
        # print(G)
        return G

    # 查询俩点间的距离
    def query(self, src, dest):
        #src_idx = self.index.get(src,None)
        #dest_idx = self.index.get(dest,None)
        #if src_idx and dest_idx:
        #    src_list = src_idx.get("backward",None)
        #    dest_list = dest_idx.get("forward",None)

        src_list = self.index[src]["backward"]
        dest_list = self.index[dest]["forward"]
        i = 0
        j = 0
        result = max_length
        
        # 构建好的index里label是按照nodes_list里的节点顺序排序的，所以可以这么写
        while i < len(src_list) and j < len(dest_list):
            if (src_list[i][0] == dest_list[j][0] and result > src_list[i][1] + dest_list[j][1]):
                result = src_list[i][1] + dest_list[j][1]
            elif self.vertex_order[src_list[i][0]] > self.vertex_order[dest_list[j][0]]:
                i += 1
            else:
                j += 1
        # print(result)
        #for (src, s_dist) in src_list:
        #print(src_list)
        #print(dest_list)
        return result

    # 加载图的Index
    def load_index(self, index_file_path):
        f = open(index_file_path, 'r')
        data = f.readlines()
        # print(f"data:{data}")
        result = json.loads(data[0])
        # print(f"result:{result}")
        self.vertex_order = json.loads(data[1])
        f.close()
        return result


    def gen_test_order(self):
        result = {}
        # 获得节点的number
        nNodes = len(self.graph.nodes())
        # print(f"nodes:{self.graph.nodes}")
        # print(f"nodes_num:{nNodes}")
        for idx, v in enumerate(self.graph.nodes()):
            result[v] = nNodes - idx
            # print(f"v:{v}")
            # print(f"result[v]:{result[v]}")
        # result['c'] = 6
        # result['d'] = 5
        # result['e'] = 4
        # result['f'] = 3
        # result['a'] = 2
        # result['b'] = 1
        return result

    def gen_random_order(self):
        result = {}
        nNodes = len(self.graph.nodes())
        # nodes_list为节点的数组
        nodes_list = list(self.graph.nodes())
        # print(nodes_list)
        random.shuffle(nodes_list)
        # print(nodes_list)
        for idx, v in enumerate(nodes_list):
            result[v] = nNodes - idx
        # print(result)
        return result

    def gen_degree_base_order(self):
        result = {}
        # print(self.graph.nodes())
        nNodes = len(self.graph.nodes())
        nodes_list = list(sorted(self.graph.degree, key=lambda x: x[1], reverse=True))
        # print(f"degree:{self.graph.degree}\n")
        for idx, v in enumerate(nodes_list):
            result[v[0]] = nNodes - idx
        #    print(v[0], result[v[0]])
        # print(result)
        return result

    def gen_closeness_base_order(self):
        result = {}
        nNodes = len(self.graph.nodes())
        nodes_list = nx.closeness_centrality(self.graph)
        # print(f"closeness:{nodes_list}")
        nodes_list = list(sorted(nodes_list.items(), key=lambda item:item[1], reverse = True))
        # print(f"nodes_list:{nodes_list}")
        for idx, v in enumerate(nodes_list):
            # print(f"idx:{idx}, v:{v}")
            result[v[0]] = nNodes - idx
            # print(f"result:{result}")
        # print(result)
        return result

    def gen_betweeness_base_order(self):
        result = {}
        nNodes = len(self.graph.nodes())
        nodes_list = nx.betweenness_centrality(self.graph, weight="weight")
        # print(f"betweenness:{nodes_list}")
        nodes_list = list(sorted(nodes_list.items(), key=lambda item:item[1], reverse = True))
        # print(f"nodes_list:{nodes_list}")
        for idx, v in enumerate(nodes_list):
            # print(f"idx:{idx}, v:{v}")
            result[v[0]] = nNodes - idx
            # print(f"result:{result}")
        # print(result)
        return result

    '''
        Function: gen_2_hop_base_order
        description: 按照我们定义的算法找出2_hop点重新进行排序
    '''
    def gen_2_hop_base_order(self):
        result = {}
        count_result = {}
        nNodes = len(self.graph.nodes())
        nodes_list = list(self.graph.nodes())
        # print(f"nodes_list:{nodes_list}")
        for node in nodes_list:
            count_result[node] = 0
        self.index = self.load_index(index_file_path)
        # print(count_result)
        # print(f"index:{self.index}")
        for src in nodes_list:
            for dest in nodes_list:
                hop_list = self.gen_hop_node(src,dest)
                for hop_node in hop_list:
                    count_result[hop_node]+=1
        # print(f"count_result:{count_result}")
        # print("test!test!")
        # 排序
        nodes_list = list(sorted(count_result.items(), key=lambda item:item[1], reverse = True))
        # print(f"nodes_list:{nodes_list}")
        # 将count_result写入文件
        fileName = "./idx_list/"+ self.map_file_name+ "_2_hop_node_count.idx"
        f = open(fileName, 'w')
        write_data = json.dumps(nodes_list)
        f.write(write_data)
        f.close()

        for idx, v in enumerate(nodes_list):
            result[v[0]] = nNodes - idx
        return result
    

    '''
        Function: gen_hop_node
        description: 得到2_hop点， 即俩个label的交点，参考了query函数的写法
    '''
    def gen_hop_node(self, src, dest):
        src_list = self.index[src]["backward"]
        dest_list = self.index[dest]["forward"]
        i = 0
        j = 0
        result = self.query(src, dest)
        if ( result == 1 or result == 0) :
            return []
        hop = src
        hop_list = []
        while i < len(src_list) and j < len(dest_list):
            if (src_list[i][0] == dest_list[j][0] and result == src_list[i][1] + dest_list[j][1]):
                # result = src_list[i][1] + dest_list[j][1]
                hop = src_list[i][0]
                hop_list.append(hop)
                i += 1 
                j += 1
            elif self.vertex_order[src_list[i][0]] > self.vertex_order[dest_list[j][0]]:
                i += 1
            else:
                j += 1
        # print(f"src:{src}->dest:{dest} : {hop}")
        return hop_list

    # 生成节点的order
    def gen_order(self, mode = 0):
        # 根据输入的mode采取不同的策略构建节点的order
        # mode = 1 => sequential order（图的输入顺序）
        # mode = 2 => 随机  
        # mode = 3 => 基于degree  
        # mode = 4 => 基于betweenness     
        # mode = 5 => 基于2-hop
        if (mode == 0):
            self.vertex_order = self.gen_test_order()
        if (mode == 1):
            print("\n*************Random****************")
            self.vertex_order = self.gen_random_order()
        if (mode == 2):
            print("\n*************Degree****************")
            self.vertex_order = self.gen_degree_base_order()
        if (mode == 3):
            print("\n*************Clossness*************")
            self.vertex_order = self.gen_closeness_base_order()
        if (mode == 4):
            print("\n*************Betweenness***********")
            self.vertex_order = self.gen_betweeness_base_order()
        if (mode == 5):
            print("\n*************2-hop-based***********")
            self.vertex_order = self.gen_2_hop_base_order()
        self.vertex_order = {k: v for k, v in sorted(self.vertex_order.items(), key=lambda item: -item[1])}
        # print("vertex order: ")
        # print(self.vertex_order)
        # print("")

    # 判断是否需要剪枝
    def need_to_expand(self, src, dest, dist = -1):
        # print("nx: %s -> %s: %d" % (src, dest, v))
        our_result = self.query(src, dest)
        v = dist
        # print("pll: %s -> %s: %d" % (src, dest, our_result))
        if (our_result <= v):
            return False
        return True

    # 基于Order进行BFS构建Index
    def build_index(self, order_mode = 0, map_file_name = ""):
        # 构建Order并记录时间      
        start_time_order = time.time()
        self.gen_order(order_mode)
        print("finish generating order")
        print(f"Time cost : {(time.time()-start_time_order):.4f}")
        print("***********************************")

        # print(f"order::{self.vertex_order}")

        # 构建BFS_num记录每轮BFS遍历的节点个数
        BFS_num_list = {}
        for index, node in enumerate(self.vertex_order):
            BFS_num_list[node] = 0
        # print(f"BFS_num_list:{BFS_num_list}")

       
        self.index = {}
        has_process = {}
        pq = Q.PriorityQueue()
        # print(f'pq:{pq}\n')
        for v in self.graph.nodes():
            self.index[v] = {"backward": [], "forward": []}
            has_process[v] = False
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
                if (has_process[src] or self.vertex_order[cur_node] < self.vertex_order[src] or not self.need_to_expand(cur_node, src, cur_dist)):
                    # print(f"self.vertex_order[cur_node]:{self.vertex_order[cur_node]}")
                    # print(f'src:{src}')
                    
                    has_process[src] = True
                    continue
                count+=1
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
            BFS_num_list[cur_node] = count
            count = 0
            # print(BFS_num_list)
            # print(f"cur_node:{cur_node}")
            # print("")
        # 将结果写入pll.idx
        self.write_BFS_num_list(map_file_name,BFS_num_list)
        self.write_index(map_file_name)
        return self.index, start_time_BFS


    # 分别构建了forward和backward是因为 图是有向图
    def build_forward_index(self, cur_node):
        has_process = {}
        pq = Q.PriorityQueue()
        pq.put((0, cur_node))
        for k in self.graph.nodes():
            has_process[k] = False
        while (not pq.empty()):
            cur_dist, src = pq.get()
            # print("[Forward] Pop: (%s %d)"%(src,cur_dist))
            if (has_process[src] or self.vertex_order[cur_node] < self.vertex_order[src] or not self.need_to_expand(cur_node, src, cur_dist)):
                has_process[src] = True
                continue
            has_process[src] = True
            self.index[src]["forward"].append((cur_node, cur_dist))
            edges = self.graph.out_edges(src)
            # print(src)
            # print(edges)
            for _, dest in edges:
                weight = self.graph.get_edge_data(src, dest)['weight']
                if (has_process[dest]):
                    continue
                pq.put((cur_dist + weight, dest))
                # print("[Forward] Push: (%s, %d)"%(dest, cur_dist + weight))

    def build_backward_index(self, cur_node):
        has_process = {}
        pq = Q.PriorityQueue()
        pq.put((0, cur_node))
        for k in self.graph.nodes():
            has_process[k] = False
        while (not pq.empty()):
            cur_dist, src = pq.get()
            # print("[Backward] Pop: (%s %d)"%(src,cur_dist))
            if (has_process[src] or self.vertex_order[cur_node] < self.vertex_order[src] or not self.need_to_expand(src, cur_node, cur_dist)):
                continue
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
                # print("[Backward] Push: (%s, %d)"%(dest, cur_dist + weight))

    def build_index_multi_thread(self, order_mode = 0):
        self.gen_order(order_mode)
        self.index = {}
        for v in self.graph.nodes():
            self.index[v] = {"backward": [], "forward": []}

        nNode = len(self.graph.nodes())
        for i, order_item in enumerate(self.vertex_order.items()):
            cur_node = order_item[0]
            print("Caculating %s (%d/%d)... " % (cur_node, i, nNode))   
            forward_thread = threading.Thread(target=self.build_forward_index, args=(cur_node,))
            backward_thread = threading.Thread(target=self.build_backward_index, args=(cur_node,))
            forward_thread.start()
            backward_thread.start()
            forward_thread.join()
            backward_thread.join()
            # print("")
        self.write_index()
        return self.index

    def build_index_multi_process(self, order_mode = 0):
        self.gen_order(order_mode)
        self.index = {}
        for v in self.graph.nodes():
            self.index[v] = {"backward": [], "forward": []}

        nNode = len(self.graph.nodes())
        for i, order_item in enumerate(self.vertex_order.items()):
            cur_node = order_item[0]
            print("Caculating %s (%d/%d)... " % (cur_node, i, nNode))
            forward_process = multiprocessing.Process(target=self.build_forward_index, args=(cur_node,))
            backward_process = multiprocessing.Process(target=self.build_backward_index, args=(cur_node,))
            forward_process.start()
            backward_process.start()
            forward_process.join()
            backward_process.join()
            # print("")
        self.write_index()
        return self.index
    
    def validation(self, times = 10):
        node_list = list(self.graph.nodes())
        nx_times = 0.0
        pll_times = 0.0
        pass_cases = 0
        # print(node_list)
        for _ in range(times):
            src = random.choice(node_list)
            dest = random.choice(node_list)
            print("Testing %s -> %s:" % (src, dest))
            start_time = time.time()
            try:
                nx_result = nx.dijkstra_path_length(self.graph, source=src, target=dest, weight="weight")
            except:
                nx_result = max_length
            interval_time = time.time()
            my_result = self.query(src, dest)
            end_time = time.time()
            print("nx: %d, time: %f" % (nx_result, interval_time - start_time))
            print("pll: %d, time: %f" % (my_result, end_time - interval_time))
            nx_times += interval_time - start_time
            pll_times += end_time - interval_time
            if (my_result == nx_result):
                pass_cases += 1

        print("Total Test Times: %d" % times)
        print("Networkx Average Time: %f" % (nx_times / times))
        print("PLL Average Time: %f" % (pll_times / times))
        print("Pass Cases: %d/%d" % (pass_cases, times))
        return 0
    
def usage(argv = []):
    print("Usage: python pll.py [ build | query | test ]")

# Function build: build order
def build(argv):
    # default parameter
    print(argv)
    map_file_name = ""
    order_mode = 0
    is_multi_process = False
    help_msg = "python pll.py build -i [input_file] -o [order_mode] -m(use multi-process)"
    try:
        # getopt是一个处理命令行参数的函数（读取并解析用户输出的命令）
        options, args = getopt.getopt(argv, "hi:o:m", ["help", "input=", "order_mode=", "multi_process"])
        for name, value in options:
            if name in ("-h", "--help"):
                print(help_msg)
                return 2
            if name in ("-i", "--input"):
                map_file_name = value
            if name in ("-o", "--order_mode"):
                order_mode = int(value)
            if name in ("-m", "--multi_process"):
                is_multi_process = True
    except:
        print(help_msg)
        return 2
    
    if (map_file_name == ""):
        print(help_msg)
        return 2

    start_time = time.time()
    pll = PrunedLandmarkLabeling(map_file_name, order_mode, False, is_multi_process)
    print(f"Total time: {(time.time() - start_time):.4f}" )
    return 0

def query(argv):
    help_msg = "python pll.py query -s [src_vectex] -t [target_vectex]"
    src_vertex = ""
    target_vertex = ""
    try:
        options, args = getopt.getopt(argv, "hs:t:", ["help", "src=", "target="])
        for name, value in options:
            if name in ("-h", "--help"):
                print(help_msg)
                return 2
            if name in ("-s", "--src"):
                src_vertex = value
            if name in ("-t", "--target"):
                target_vertex = value
    except:
        print(help_msg)
        return 2

    start_time = time.time()
    pll = PrunedLandmarkLabeling()
    pll.query(src_vertex, target_vertex)
    print(f"Total time: {(time.time() - start_time)}")
    return 0


def test(argv):
    help_msg = "python pll.py test -t [times] -m [map_file]"
    times = 10
    map_file_name = ""
    try:
        options, args = getopt.getopt(argv, "ht:m:", ["help", "times=", "map_file="])
        for name, value in options:
            if name in ("-h", "--help"):
                print(help_msg)
                return 2
            if name in ("-t", "--target"):
                times = int(value)
            if name in ("-m", "--map_file"):
                map_file_name = value
    except:
        print(help_msg)
        return 2

    if (map_file_name == ""):
        print(help_msg)
        return 2

    pll = PrunedLandmarkLabeling(map_file_name, 0, True)
    pll.validation(times)
    return 0

action = {
        "build": build,
        "query": query,
        "test": test,
        "help": usage
    }

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        usage()
        sys.exit(2)
    
    action = {
        "build": build,
        "query": query,
        "test": test,
        "help": usage
    }
    print(sys.argv)
    sys.exit(action.get(sys.argv[1], usage)(sys.argv[2:]))


    

