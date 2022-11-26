import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"]
x1 = [4534,6523,10591]
x2 = [7864,12152,24746]
hop_index_time = [10.0443,15.6714,28.4485]

betweenness_index_time = [26.3426,49.0946,91.3471]

hop_index_size = [979.88,1278.7863,1859.2409]

betweenness_index_size = [1909.002,2864.7133,4398.1588]

hop_query_time = [3.0658,3.2094,2.974]

betweenness_query_time = [5.5018,6.5435,6.451]

plt.subplot(2,3,1)
plt.plot(x1,hop_index_time,x1,betweenness_index_time,marker='*')
plt.xlabel("nodes")
plt.ylabel("index_time")
plt.legend(['2-hop-base','betweenness'])
plt.title("标签构建时间-随nodes")

plt.subplot(2,3,2)
plt.plot(x1,hop_index_size,x1,betweenness_index_size,marker='*')
plt.xlabel("nodes")
plt.ylabel("index_size")
plt.legend(['2-hop-base','betweenness'])
plt.title("标签大小-随nodes")

plt.subplot(2,3,3)
plt.plot(x1,hop_query_time,x1,betweenness_query_time,marker='*')
plt.xlabel("nodes")
plt.ylabel("query_time")
plt.legend(['2-hop-base','betweenness'])
plt.title("查询时间-随nodes")

plt.subplot(2,3,4)
plt.plot(x2,hop_index_time,x2,betweenness_index_time,marker='*')
plt.xlabel("edges")
plt.ylabel("index_time")
plt.legend(['2-hop-base','betweenness'])
plt.title("标签构建时间-随edges")

plt.subplot(2,3,5)
plt.plot(x2,hop_index_size,x2,betweenness_index_size,marker='*')
plt.xlabel("edges")
plt.ylabel("index_size")
plt.legend(['2-hop-base','betweenness'])
plt.title("标签大小-随edges")

plt.subplot(2,3,6)
plt.plot(x2,hop_query_time,x2,betweenness_query_time,marker='*')
plt.xlabel("edges")
plt.ylabel("query_time")
plt.legend(['2-hop-base','betweenness'])
plt.title("查询时间-随edges")


plt.show()