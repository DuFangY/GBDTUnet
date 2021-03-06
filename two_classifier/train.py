import csv

import numpy as np
import pandas as pd
import pickle
import joblib
from Gbdt import Tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
class AllRules1():
    def __init__(self,rules,from_trees):
        self.rules = rules
        self.from_trees = from_trees
        self.num = 1 #相同规则数量初始化为1

    def __str__(self):
        return self.rules  #这里返回规则

    def __repr__(self):
        return self.__str__()

def fit_unet_onlyGDBT():
    """
    训练数据
    :return:
    """
    data = pd.read_csv("1.csv")
    y = data.zLabel.values
    X = data.drop("zLabel", axis=1)
    features = X.columns
    X = X.values
    X = X.astype(np.float32)
    X = StandardScaler().fit_transform(X)  #标准化
    X = pd.DataFrame(X)
    X = X.values
    #
    # gdbt_tree = Tree()
    # gdbt_tree.fit(X, y, feature_names=features)
    # path = './GBDT_only.pkl'
    # joblib.dump(gdbt_tree,path)

    path = './GBDT_only.pkl'
    gbdt_tree = joblib.load(path)
    """
    测试数据
    """
    data = pd.read_csv("1_test.csv")
    data = data.fillna(0)
    y = data.zLabel.values
    X = data.drop("zLabel", axis=1)
    features = X.columns.tolist()
    X = X.values
    X = X.astype(np.float64)
    X = StandardScaler().fit_transform(X)  #标准化
    X = pd.DataFrame(X)
    X = X.values
    #预测并提取规则以及重要性
    pre = gbdt_tree.predict(X)
    accuracy = accuracy_score(y, pre)
    print("\n分类准确度为： ",accuracy)
    # allRules = gbdt_tree.rule_ensemble.rulesAll  #所有规则
    # temp = allRules.copy()
    # temp.insert(0, 'index')
    # X1Rules = gbdt_tree.X1_rules
    # X1Index = gbdt_tree.flag_1 #正确预测为样本1的索引

    """
    规则写入csv
    """
    # with open(r'./satisfy_rules1.csv',"a",newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(temp)
    #     for index,item in enumerate(X1Rules):
    #         value = item.tolist()
    #         value.insert(0,X1Index[index])
    #         writer.writerow(value)
    #     file.close()
    # print("标签1样本规则提取完成 ！")
    # exRules = extractRules1(r'./satisfy_rules1.csv',allRules)
    # """
    # 将规则按照样本满足数量由上到下排序
    # """
    # for i in range(len(exRules)-1):
    #     for j in range(i+1,len(exRules)):
    #         if(exRules[i].num < exRules[j].num):
    #             val = exRules[i]
    #             exRules[i] = exRules[j]
    #             exRules[j] = val
    # rulesMergeRank = []
    # rulesMergeRank_str = []
    # for oneRules in exRules:
    #     origin_numbers = oneRules.num #没有规则去重前满足此规则样本个数
    #     origin_fromTree = oneRules.from_trees #没有规则去重前满足的规则来自哪几棵树
    #     mergeRules = mergingRules(str(oneRules))
    #     mergeRules = " & ".join(mergeRules)
    #     if(mergeRules in rulesMergeRank_str): #不同样本规则重复
    #         index = rulesMergeRank_str.index(mergeRules) #重复规则的索引
    #         rulesMergeRank[index].num += origin_numbers
    #         """
    #         更新来自哪几棵树
    #         """
    #         fromTreeTemp1 = origin_fromTree
    #         fromTreeTemp2 = rulesMergeRank[index].from_trees
    #         treeFlag = set()
    #         treeFlag.update(fromTreeTemp1)
    #         treeFlag.update(fromTreeTemp2)
    #         treeFlag = list(treeFlag)
    #         treeFlag.sort()
    #         rulesMergeRank[index].from_trees = treeFlag
    #     else:
    #         ruleMergeObject = AllRules1(mergeRules,origin_fromTree)
    #         ruleMergeObject.num = origin_numbers
    #         rulesMergeRank.append(ruleMergeObject)
    #         rulesMergeRank_str.append(mergeRules) #此列表只加入str 为了后续比较是否列表有重复
    #
    # joblib.dump(rulesMergeRank,'rules1.pkl')
    #
    # label1Rules(gdbt_tree) #满足规正确预测为1标签的样本规则集合
    #



def extractRules1(path,allRules):
    data = pd.read_csv(path)
    id = data[data.columns[0]] #此样本在所有样本中的索引
    data = data[data.columns[1:]].values

    Rules = []
    for i in range(data.shape[0]):
        print("样本 %s 正在提取满足的规则..." % (i))
        oneSample = []  #存储每个样本每棵树符合条件的规则
        for j in range(data.shape[1]):
            if(data[i][j] == 1 and allRules[j].friedman_mse <=10):
                oneSample.append(allRules[j])
        if(len(oneSample) == 0):
            continue
        Rules.append(oneSample) #[ [第一个样本规则],[第二个样本规则],[  ,    ,   ]......  ]
    rulesRank = []
    rulesRank_str = []
    for ruleList in Rules:
        #每一个样本的规则集
        Tree = [] #记录每个样本规则来自的树的个数
        decidePath=''
        for decpath in ruleList:
            Tree.append(decpath.fromTree) #将此样本所使用的树记录下来
            decidePath += (str(decpath) + "&")
        decidePath = decidePath[0:-1] #去除最后一个&符号
        Tree.sort()
        allR = AllRules1(decidePath,Tree)#每个样本记录缩减后重要性高的路径
        if(allR.rules in rulesRank_str):
            index = rulesRank_str.index(allR.rules) #找出index
            rulesRank[index].num += 1
        else:
            rulesRank.append(allR)
            rulesRank_str.append(allR.rules) #这里将对象实例化去掉，以便后续直接比较字符
    return rulesRank

def mergingRules(rules):
    feature_name = []  #规则涉及到的特征名
    op = [] # 规则判断符号 0代表小于等于 1代表大于
    judgeValue = [] #规则涉及到的阈值
    rules = rules.split("&")
    for i in range(len(rules)):
        rules[i] = rules[i].strip(' ') #去除因为之前操作所得到前后的空格
    rules.sort()
    mergingAfterRules=[]
    for rule in rules:
        rule_split = rule.split(" ")
        feature_name.append(rule_split[0])
        #0代表小于等于 1代表大于
        if rule_split[1] == '<=':
            op.append(0)
        else:
            op.append(1)
        judgeValue.append(float(rule_split[2]))
    i = 0
    while(i< len(feature_name)):
        j = i+1 #特征名称前后比较
        count = 0 #计算规则涉及特征重复的次数
        while(1):
            if(j == len(feature_name)):
                break
            if(feature_name[i] != feature_name[j]):
                break
            if(feature_name[i] == feature_name[j]):
                count += 1 #如果规则相等 规则重复的次数加1
                j = j+1 #后边比较的指针后移
        if count > 0:
            """
            这里为合并相同特征规则
            """
            k = i #从第一个比较的规则处开始进行合并
            min_up = 99999 #最小上界
            max_down = -99999 #最大下界
            flag_0 = -1 #标志是否规则中有<=
            flag_1 = -1 #标志是否规则中有>
            while(k<j):
                if(op[k] == 0): # <=
                    flag_0 = 1
                    if(judgeValue[k] < min_up):
                        min_up = judgeValue[k]
                else: # >
                    flag_1 = 1
                    if(judgeValue[k] > max_down):
                        max_down = judgeValue[k]
                k += 1

            if(flag_0 == 1): #标志有<= 此处添加合并后<=的规则
                new_rule = feature_name[i] + " <= " + str(min_up)
                mergingAfterRules.append(new_rule)
            if(flag_1 ==1): #标志有> 此处添加合并后>的规则
                new_rule = feature_name[i] + " > " + str(max_down)
                mergingAfterRules.append(new_rule)
            i = j-1 # 由于前边还需要 i+1 所以此时i 需要等于 j(与前边特征第一次不相等的)  的前一个
        else: #前后两条规则涉及到的特征不相同，前一条直接进入修改合并后的规则列表
            mergingAfterRules.append(rules[i])

        i += 1
    return mergingAfterRules

def label1Rules(gdbt_tree):  #满足规正确预测为1标签的样本规则集合
    allDecidePath = []
    allLabel1Rules = set()  # 满足规正确预测为1标签的样本规则集合
    for i in range(gdbt_tree.X1_rules.shape[0]):  # 遍历每一个正确预测为标签1的样本
        print("样本%s:" % (i))
        decide_path = ''
        for j in range(gdbt_tree.X1_rules.shape[1]):  # 遍历每一条规则，看此样本是否满足，并且输出
            if gdbt_tree.X1_rules[i][j] == 1:
                decide_path = decide_path + str(gdbt_tree.rule_ensemble.rulesAll[j]) + " & "
        decide_path = decide_path[0:-1]
        rules_content = decide_path.split(" & ")
        print('此样本规则长度: ', len(rules_content))
        oneDecidePath = set()  # 每个样本会清空一次，目的是为了删除重复规则
        oneDecidePath.update(rules_content)  # 每一个样本的决策路径
        temp_list = list(oneDecidePath)  # 这里已经去除重复的规则了
        temp_list.sort()  # 为了对每个样本决策路径进行提取，先对每个样本去重后决策路径进行排序
        print("去除重复后长度为: ", len(temp_list))
        temp_list = mergingRules(temp_list)  # 将部分类似规则合并
        temp_list.sort()
        print("合并冗余规则后长度为: ", len(temp_list))
        allDecidePath.append(temp_list)  # 将整理去重后的决策路径加入决策路径规则中[[],[]...]
        allLabel1Rules.update(temp_list)  # 添加正确预测为标签1的规则[, , ,]
        print(temp_list)
        print("_____________________________________________________________________________________")
    allLabel1Rules = list(allLabel1Rules)
    allLabel1Rules.sort()
    temp = allLabel1Rules
    temp.insert(0,'index')

    with open(r'./satisfy_rules1.csv',"a",newline='') as file:
        writer = csv.writer(file)
        writer.writerow(temp)
        count = -1
        for train_rules in allDecidePath: #遍历每个正确预测为1的样本的决策结点规则
            count += 1
            rules1 = [0 for n in range(len(temp))]
            for nodeJudge in train_rules:
                rules1[temp.index(nodeJudge)] = 1
            rules1[0]=gdbt_tree.flag_1[count] #此样本的索引
            writer.writerow(rules1)
        file.close()
        print("标签1决策节点写入完成！")



if __name__ == '__main__':
    fit_unet_onlyGDBT()