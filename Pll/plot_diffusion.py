import getopt
import sys
import json
import matplotlib.pyplot as plt

def entrance():
    file_name = fetch_file_name()
    data = fetch_data_from_idx(file_name)
    data_process(data) 

def fetch_file_name():
    help_msg = "python plot_diffusion.py -i [input file]"
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
    # print(raw_data)
    formated_data = eval(raw_data)
    # print(formated_data)
    return formated_data

def data_process(data):
    # print(data)
    data[i]
    plot_data = {}
    key_list_data = list(data.keys())
    print("*************************")
    print("*************************")
    print("*************************")
    print("*************************")
    print("*************************")

    for i in range(len(data)):
        plot_data[i] = data[key_list_data[i]]
    x = list(plot_data.keys())
    y = list(plot_data.values())
    print(x)
    print(y)
    plt.scatter(x,y,s=1)
    plt.show()
        

    # print(plot_data)

entrance()