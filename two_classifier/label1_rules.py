import re
import pandas as pd
from itertools import combinations
from sklearn.metrics import accuracy_score
import time
from train import AllRules1
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

def test_label1Rules(path):
    rules = joblib.load(path)
    # test_data = pd.read_csv(r'./1_test.csv')
    test_data = pd.read_csv(r'./1.csv')
    # test_data = test_data.loc[test_data['zLabel'] == 1]
    allColumns = (test_data.columns.values).tolist()[1:]
    label = (test_data[test_data.columns[0:1]].values).tolist()
    data = test_data[test_data.columns[1:]]
    data = data.astype(np.float32)
    data = StandardScaler().fit_transform(data)  #标准化
    data = data.tolist()
    predict = []
    count = 0
    path = './GBDT_only.pkl'
    gbdt_tree = joblib.load(path)
    a =gbdt_tree.predictTrue1
    # temp = set()

    for samples in data:
        count += 1
        print("样本%s(%s)判断中..."%(count,len(data)))
        for rule in rules:
            flag = 0  # 是否结束此样本判断的标志
            ruleList = str(rule).split(' & ')
            for node in ruleList:
                feature = node.split(" ")[0] #需要比较的特征
                op = node.split(" ")[1] #判断符号
                judgeValue = float(node.split(" ")[2]) #判断值
                sampleValue = float(samples[allColumns.index(feature)]) #找到需要比较的特征
                if(op == "<="):
                    if(sampleValue > judgeValue):
                        flag = 1  # 样本不符合此条规则
                        break
                if(op == ">"):
                    if(sampleValue <= judgeValue):
                        flag = 1  # 样本不符合此条规则
                        break
            if(flag == 0): #经过一条规则，全部满足则跳出规则遍历判断
                predict.append(1)
                print('here')
                # if (label[count - 1][0] == 1):
                #     temp.update([rule])
                break
        if(flag == 1): #所有规则不满足此样本，预测值为0
            predict.append(0)
    # joblib.dump(temp, 'R1.pkl')
    accuracy = accuracy_score(label, predict)
    print("\n分类准确度为： ",accuracy)
if __name__ == '__main__':
    test_label1Rules(r'./rules1.pkl')

# def extractLabel1Rule(path):
#     data = pd.read_csv(path,error_bad_lines=False)
#     columns = data.columns[1:]
#     data = data[data.columns[1:]].values
#     with open(r'./satisfy_rules1.txt','w') as file:
#         for nodeJudge in range(data.shape[1]):
#             count1 = (data[:,nodeJudge].tolist()).count(1)
#             percentage = count1 / data.shape[0]
#             if percentage >=0.75:
#                 file.writelines(columns[nodeJudge]+'\n')
#     print("标签1规则节点已经提取完成！")
#
# def validLabel1Rules(path):
#     accuracys = []
#     with open(path, 'r') as file:
#         content_all = [val.replace('\n',"" ) for val in file.readlines()]
#         # for val in content_all:
#         #     temp = re.findall(r'(.*?) ',val)[0]
#         #     feature.append(temp)
#     print("标签1得到的规则节点共%s个"%(len(content_all)))
#     print(content_all)
#     content_combine = []
#     time1 = time.clock()
#     for i in range(29):
#         i += 1
#         content_combine.append(combine(content_all,i,29))
#     time2 = time.clock()
#     allRuleCombineTime = time2-time1 #规则组合时间
#     datas = pd.read_csv(r'1.csv')
#     Label = datas['zLabel']
#     time3 = time.clock()
#     for c in content_combine:
#         for content in c:
#             """
#             读取测试 测试文件
#             """
#             feature = []
#             for val in content:
#                 temp = re.findall(r'(.*?) ',val)[0]
#                 feature.append(temp)
#             data = datas[feature].values #这里是按照特征列表传进去的顺序读取的，所以和节点判断规则中的顺序一致
#             predict = [0 for n in range(data.shape[0])] #初始化根据规则预测结果
#             for i in range(data.shape[0]): #遍历每一个测试样本
#                 test = data[i,:].tolist()
#                 count = -1
#                 satisify = 1 #标志此样本按照得到规则是否全部满足，若全部满足则此样本为标签1
#                 for nodeJudge in test:
#                     count += 1
#                     node = content[count]
#                     flag = re.findall(r'> (.*?)$',node) #规则节点值
#                     if(len(flag)!=0): #此规则为判断是否 nodeJudge＞规则节点值
#                         flag=float(flag[0])
#                         if(nodeJudge <= flag): #不满足此节点规则
#                             satisify = -1
#                             break
#                     else:  #此规则为判断是否 nodeJudge<=规则节点值
#                         flag = re.findall(r'<= (.*?)$', node)
#                         flag=float(flag[0])
#                         if(nodeJudge > flag): #不满足此节点规则
#                             satisify = -1
#                             break
#                 if(satisify == 1): #全部满足所提规则，预测结果为标签1
#                     predict[i] = 1
#             accuracy = accuracy_score(Label, predict)
#             accuracys.append(accuracy)
#             print("规则： ",content," 下标签1预测准确度为：%s"%accuracy)
#     time4 = time.clock()
#     print("所有预测精度为：%s"%accuracys)
#     print("规则组合所使用的时间为%s"%allRuleCombineTime)
#     print("组合规则精确度遍历判别所使用的时间为%s" %(time4-time3))
#
# def combine(temp_list, n,N):
#     '''根据n获得列表中的所有可能组合（n个元素为一组）'''
#     temp_list2 = []
#     for c in combinations(temp_list, n):
#         temp_list2.append(c)
#     print("%s/%s个规则节点已经组合完成！"%(n,N))
#     return temp_list2
#
#
# if __name__ == '__main__':
#     # extractLabel1Rule(r'./satisfy_rules1.csv')
#     validLabel1Rules(r'./satisfy_rules1.txt')