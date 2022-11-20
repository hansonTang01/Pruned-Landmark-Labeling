import getopt
import sys
import json
import matplotlib.pyplot as plt

def entrance():
    file_name = fetch_file_name()
    data = fetch_data_from_idx(file_name)
    data_process(data)

def fetch_file_name():
    help_msg = "python plot_histogram.py -i [input file]"
    try:
        options, args = getopt.getopt(sys.argv[1:], "i:", ["input="])
        file_name = options[0][1]
        return file_name 
    except:
        print(help_msg)
        exit()
        
def fetch_data_from_idx(file_name):
    f = open(file_name,'r')
    raw_data = f.readlines()
    formated_data = eval(raw_data[0])
    # print(formated_data)
    return formated_data

def data_process(data):
    test_data=[]
    plot_data = {}
    for item in data:
        print(item)
        test_data.append(item[1])
        plot_data[str(item[1])] = 0
    # plt.hist(test_data,bins=100)
    # plt.title("2-hop-analyzeda")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()
    for item in data:
        plot_data[str(item[1])]+=1
        
    print(plot_data)
    # print(plot_data)

entrance()