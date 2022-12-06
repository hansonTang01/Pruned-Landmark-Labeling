from datetime import datetime
import pll
import sys
import getopt
import numpy as np
from random import randint
import time
import fetchData 
import pandas as pd
import xlsxwriter 
#程序入口
def begin():
    map_file_name = fetch_map_name()
    BFS_list, Index_list, Query_list,build_order_time_list = fetchData.entrance(map_file_name)
    total_time_list = [BFS_list[i] + build_order_time_list[i] for i in range(len(BFS_list))]
    df = data2pandas(BFS_list,Index_list,Query_list,build_order_time_list, total_time_list)
    print(df)
    dataFrame2excel(df,map_file_name)

# 从命令行参数中获取mapName
def fetch_map_name():
    help_msg = "python data2Excel.py -i [input_file]"
    try:
        options, args = getopt.getopt(sys.argv[1:], "i:", ["input="])
        map_file_name = options[0][1]
        return map_file_name 
    except:
        print(help_msg)
        exit()

# 通过pandas将list中的data插入Dataframe
def data2pandas(BFS_list,Index_list,Query_list,build_order_time_list, total_time_list):
    df = pd.DataFrame(columns=["Degree","betweenness",'betweenness-2-hop-count'])
    df.loc[len(df.index)] = BFS_list
    df.loc[len(df.index)] = Index_list
    df.loc[len(df.index)] = Query_list
    df.loc[len(df.index)] = build_order_time_list
    df.loc[len(df.index)] = total_time_list
    df.rename(index = {0:"indexing_time",1:"avg_label_size",2:"query_time_100000",3:"build_order_time",4:"each_total_time"},inplace=True)

    # print(df)
    return df

# 将dataframe转为excel保存
def dataFrame2excel(df, map_file_name):
    excel_file_name = "./excel_list/"+map_file_name+".xlsx"
    # excel_file_name = "test2.xlsx"

    writer = pd.ExcelWriter(excel_file_name, engine='xlsxwriter')

    df.to_excel(writer,'Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    worksheet.set_column('A:G', 25)
    total_time = df.loc['each_total_time',"Degree"] + df.loc['each_total_time',"betweenness"] + df.loc['each_total_time',"betweenness-2-hop-count"]
    print(f"total_time: {total_time}")
    worksheet.write(7,0,total_time)
    writer.save()
    # df.to_excel(excel_file_name,            # 路径和文件名
    #         sheet_name='sheet1',     # sheet 的名字
    #         float_format='%.4f',  # 保留两位小数
    #         na_rep='0') 
    return 0
if __name__ == "__main__":
    begin()