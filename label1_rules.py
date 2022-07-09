import re

import pandas as pd
def extractLabel1Rule(path):
    data = pd.read_csv(path)
    columns = data.columns[1:]
    data = data[data.columns[1:]].values
    with open(r'./satisfy_rules1.txt','w') as file:
        for nodeJudge in range(data.shape[1]):
            count1 = (data[:,nodeJudge].tolist()).count(1)
            percentage = count1 / data.shape[0]
            if percentage >=0.80:
                file.writelines(columns[nodeJudge]+'\n')
    print("标签1规则节点已经提取完成！")

def validLabel1Rules(path):
    feature = []
    with open(path, 'r') as file:
        content = [val.replace('\n',"" ) for val in file.readlines()]
        for val in content:
            temp = re.findall(r'(.*?) ',val)[0]
            feature.append(temp)
    print("标签1得到的规则节点共%s个"%(len(content)))
    print(content)
    """
    读取测试 测试文件
    """
    data = pd.read_csv(r'test.csv')
    Label = data['zLabel']
    data = data[feature].values #这里是按照特征列表传进去的顺序读取的，所以和节点判断规则中的顺序一致
    predict = [-1 for n in range(data.shape[0])] #初始化根据规则预测结果
    for i in range(data.shape[0]): #遍历每一个测试样本
        test = data[i,:].tolist()
        count = -1
        satisify = 1 #标志此样本按照得到规则是否全部满足，若全部满足则此样本为标签1
        for nodeJudge in test:
            count += 1
            flag = re.findall(r'> (.*?)$',content[count]) #规则节点值
            if(len(flag)!=0): #此规则为判断是否 nodeJudge＞规则节点值
                flag=float(flag[0])
                if(nodeJudge <= flag): #不满足此节点规则
                    satisify = -1
                    break
            else:  #此规则为判断是否 nodeJudge<=规则节点值
                flag = re.findall(r'<= (.*?)$', content[count])
                flag=float(flag[0])
                if(nodeJudge > flag): #不满足此节点规则
                    satisify = -1
                    break
        if(satisify == 1): #全部满足所提规则，预测结果为标签1
            predict[i] = 1
            print(i)
    print(Label)
    print(predict)




if __name__ == '__main__':
    extractLabel1Rule(r'./satisfy_rules1.csv')
    validLabel1Rules(r'./satisfy_rules1.txt')