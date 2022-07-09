import pandas as pd
from functools import reduce

data = pd.read_csv('./satisfy_rules1.csv')
data = data[data.columns[1:]].values
rule_applies = [0,0]
N = ((data[0,:]).tolist()).count(1)
rule_applies[0] =data[0,:]
for i in range(data.shape[0]):
    rule_applies[1] = data[i,:]
    a = reduce(lambda x, y: x * y, rule_applies)
    print((a.tolist()).count(1))
# from itertools import combinations
# def combine(temp_list, n,N):
#     '''根据n获得列表中的所有可能组合（n个元素为一组）'''
#     temp_list2 = []
#     for c in combinations(temp_list, n):
#         # flag = set()
#         # flag.update(c)
#         # flag = list(flag)
#         # if(len(c)!=n):
#         #     continue
#         temp_list2.append(c)
#     print("%s/%s个规则节点已经组合完成！"%(n,N))
#     print(temp_list2)
#     return temp_list2
#
# a = [1,2,3,4,7,9]
# c = []
# combine(a,4,3)
# # for i in range(20):
# #     i += 1
# #     c.append(combine(a, i, 20))