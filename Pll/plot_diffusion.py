import getopt
import sys
import json
import matplotlib.pyplot as plt

def entrance():
    file_name = fetch_file_name()
    data = fetch_data_from_idx(file_name)
    x, y = data_process(data) 
    plot_diffusion(x,y,file_name[11:])
# 取用户输入的idx文件名
def fetch_file_name():
    help_msg = "python plot_diffusion.py -i [input file]"
    try:
        options, args = getopt.getopt(sys.argv[1:], "i:", ["input="])
        file_name = options[0][1]
        return file_name 
    except:
        print(help_msg)
        exit()

# 从用户输入的idx文件名取数据
def fetch_data_from_idx(file_name):
    f = open(file_name,'r')
    raw_data = f.readlines()
    # print(raw_data)
    # print(raw_data)
    return raw_data

# 生成所有的x和y
def data_process(raw_data):
    # print(data)
    # print(len(data))
    x = []
    y = []
    for i in range(len(raw_data)):
        print(eval(raw_data[i]))
        x_new,y_new = generate_x_y(eval(raw_data[i]))
        x.append(x_new)
        y.append(y_new)
    return x,y

# 对每一个order对应的BFS生成x和y
def generate_x_y(data):
    plot_data = {}
    key_list_data = list(data.keys())

    for i in range(len(data)):
        plot_data[i] = data[key_list_data[i]]
    x = list(plot_data.keys())
    y = list(plot_data.values())
    return x,y

def plot_diffusion(x,y,file_name):
    
    # for i in range(len(x)):
    #     plt.scatter(x[i],y[i],s=3)
    plt.scatter(x[0],y[0],s=1)
    plt.scatter(x[1],y[1],s=1)
    plt.scatter(x[2],y[2],s=1)
    plt.scatter(x[3],y[3],s=1)
    plt.scatter(x[4],y[4],s=1)

    plt.rc('legend', fontsize=20)
    plt.legend(['degree','closeness','betweenness','2-hop','label-count-base'])
    
    plt.xlabel("x-th BFS")
    plt.ylabel("Vertices")
    plt.title(file_name)
    plt.show()
entrance()