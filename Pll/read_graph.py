'''跑哪一个就把哪个解开注释'''

'''
##################################################################################################################################
## case1: 文本文件每一行是 "v1，v2"格式  在后面添加 ",1,1"即可                                                                         ##
## 记得把pll.py和test_query.py中src, dest, dist, is_one_way = lines.split(" ")替换为src, dest, dist, is_one_way = lines.split(",") ##
##################################################################################################################################

f = open(r'bio-grid-worm.edges')   ##此处''内为文本名
lines = f.readlines()
for i in range(1,len(lines)):
    lines[i]=lines[i][0:-1]+",1,1\n"

f = open(r'bio-grid-worm.edges','w')
f.writelines(lines)
f.close()
'''

'''
################################################################
## case2: 文本文件每一行是 "v1 v2 d"格式  在后面添加 " 1"或" 0"即可  ##
################################################################

f = open(r'bio-grid-worm.edges')  ##此处''内为文本名
lines = f.readlines()
for i in range(1,len(lines)):
    lines[i]=lines[i][0:-1]+" 1\n"
    ##lines[i]=lines[i][0:-1]+" 0\n"

f = open(r'bio-grid-worm.edges','w')
f.writelines(lines)
f.close()
'''

'''
###########################################################
## case3: 文本文件每一行是 "v1 v2"格式  在后面添加 " 1 1"即可  ##
###########################################################

f = open(r'bio-grid-worm.edges')  ##此处''内为文本名
lines = f.readlines()
for i in range(1,len(lines)):
    lines[i]=lines[i][0:-1]+" 1 1\n"

f = open(r'bio-grid-worm.edges','w')
f.writelines(lines)
f.close()
'''