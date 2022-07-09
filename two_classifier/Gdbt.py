"""Linear model of tree-based decision rules
This method implement the RuleFit algorithm
The module structure is the following:
- ``RuleCondition`` implements a binary feature transformation
- ``Rule`` implements a Rule composed of ``RuleConditions``
- ``RuleEnsemble`` implements an ensemble of ``Rules``
- ``RuleFit`` implements the RuleFit algorithm
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from functools import reduce
from sklearn.ensemble import GradientBoostingClassifier
import pydotplus
from sklearn.tree import export_graphviz

class RuleCondition():
    """Class for binary rule condition
    Warning: this class should not be used directly.
    """

    def __init__(self,
                 feature_index,
                 threshold,
                 operator,
                 support,
                 count,
                 feature_name=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.feature_name = feature_name
        self.count = count

    # 出某个实例化对象时，其调用的就是该对象的 __repr__()
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature_index
        return "%s %s %s" % (feature, self.operator, self.threshold)

    def transform(self, X):
        """Transform dataset.
        Parameters
        ----------
        X: array-like matrix, shape=(n_samples, n_features)
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        # 找出所有样例中满足此条规则的样本：shape=(n_samples, 1) n个样本 1个在完整决策路径上的单规则
        if self.operator == "<=":
            res = 1 * (X[:, self.feature_index] <= self.threshold)  # 满足res 值为1， 不满足res值为0
        elif self.operator == ">":
            res = 1 * (X[:, self.feature_index] > self.threshold)  # 满足res 值为1， 不满足res值为0
        return res

    # 判断两个对象是否相等
    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.feature_index, self.threshold, self.operator, self.feature_name))

class Rule():
    """Class for binary Rules from list of conditions
    Warning: this class should not be used directly.
    """

    def __init__(self,
                 rule_conditions, prediction_value):
        self.conditions = set(rule_conditions)
        self.support = min([x.support for x in rule_conditions])  #到达此条规则的样本占比数
        self.prediction_value = prediction_value
        self.rule_direction = None

    def transform(self, X):
        """Transform dataset.
        Parameters
        ----------
        X: array-like matrix
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        # b = self.conditions
        # for d in b:
        #     c = d.transform(X)
        # condition.transform(X) 调用 RuleCondition类的transform(X)
        # condition为每一条完整决策路径的单个路径
        #所有正确预测样本是否满足预测规则 一行中为1的即为此索引正确预测样本满足此单规则
        rule_applies = [condition.transform(X) for condition in self.conditions]
        # 找出满足一条完整决策路径的样本 值为 1 的样本就满足此决策路径
        return reduce(lambda x, y: x * y, rule_applies)

    def __str__(self):
        return " & ".join([x.__str__() for x in self.conditions])  #这里返回规则

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

def extract_rules_from_tree(tree, count,feature_names=None):
    """Helper to turn a tree into as set of rules
    count 记录提取规则所属的树
    """
    rules = set()
    def traverse_nodes(node_id=0,
                       operator=None,
                       threshold=None,
                       feature=None,
                       conditions=[]):
        # 规则产生
        if node_id != 0:# 当前结点不是根节点，加入上一结点信息
            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature
            rule_condition = RuleCondition(feature_index=feature,
                                           threshold=threshold,
                                           operator=operator,
                                           support=tree.n_node_samples[node_id] / float(tree.n_node_samples[0]),
                                           count=count,
                                           #到达节点 i 的训练样本数/到达节点 0 的训练样本数
                                           #也就是数据集中满足规则rk的样本比例 sk
                                           feature_name=feature_name
                                           )  #为了将特征与大于小于号连接起来组成规则
            new_conditions = conditions + [rule_condition]
        else:
            new_conditions = []
            ## if not terminal node
            """
            1、对于节点数组left和right，-1表示叶节点
            2、对于阈值threshold，代表了当前节点选用相应特征时的分裂阈值，一般是≤该阈值时进入左子节点，否则进入右子节点，-2表示叶节点的特征阈值
            3、图上的数值是保留了3位小数的浮点数结果
            4、left数组是根节点（不含根节点本身）左子树前序遍历的节点id序列（含叶节点）
            5、right数组是根节点（不含根节点本身）的右子树前序遍历的节点id序列（含叶节点）
            6、threshold和features的顺序就是left——>right顺序
            7、feature：size类型，代表了当前节点用于分裂的特征索引，即在训练集中用第几列特征进行分裂
            8、n_node_samples：size类型，代表了训练时落入到该节点的样本总数。显然，父节点的n_node_samples将等于其左右子节点的n_node_samples之和
            """
        if tree.children_left[node_id] != tree.children_right[node_id]:
            # 如果是叶子结点左右子节点值为 -1
            feature = tree.feature[node_id]  # 当前结点特征序号
            threshold = tree.threshold[node_id]  # 当前结点阈值
            # 这里也是先序遍历结点
            left_node_id = tree.children_left[node_id]  # 当前结点的左孩子
            traverse_nodes(left_node_id, "<=", threshold, feature, new_conditions)
            new_conditions = new_conditions
            right_node_id = tree.children_right[node_id]  # 当前结点的右孩子
            traverse_nodes(right_node_id, ">", threshold, feature, new_conditions)
        else:  # a leaf node
            if len(new_conditions) > 0:
                #返回的规则是用 & 连接的
                new_rule = Rule(new_conditions, tree.value[node_id][0][0])   #实例化规则包括有 conditions(RuleCondition类中的属性) prediction_value support
                rules.update([new_rule])
            else:
                pass  # tree only has a root node!
            return None

    traverse_nodes()

    return rules

class RuleEnsemble():
    """Ensemble of binary decision rules
    This class implements an ensemble of decision rules that extracts rules from
    an ensemble of decision trees.
    Parameters
    ----------
    tree_list: List or array of DecisionTreeClassifier or DecisionTreeRegressor
        Trees from which the rules are created
    feature_names: List of strings, optional (default=None)
        Names of the features
    Attributes
    ----------
    rules: List of Rule
        The ensemble of rules extracted from the trees
    """

    def __init__(self,
                 tree_list,
                 result,
                 feature_names=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        """
        之所以是集合，是因为要将重复的规则去除，而set()又会自动去除重复的元素
        """
        self.rulesAll = set() #所有规则集合
        # self.rules_1 = set()  #每个评估器中的第一类标签
        # self.rules_others = set()  #每个评估器中的其他类标签
        self.result = result  #gdbt在训练集上的预测结果
        ## TODO: Move this out of __init__
        self._extract_rules()  #提取规则
        self.rulesAll = list(self.rulesAll)
        # self.rules_1 = list(self.rules_1)
        # self.rules_others = list(self.rules_others)


    def _extract_rules(self):
        """Recursively extract rules from each tree in the ensemble
        """
        count = 0 # 记录提取规则的树，树的序号从1开始
        for tree in self.tree_list:
            count += 1
            # tree[0].tree_ 每一个树对象
            """
            1、对于节点数组left和right，-1表示叶节点
            2、对于阈值threshold，代表了当前节点选用相应特征时的分裂阈值，一般是≤该阈值时进入左子节点，否则进入右子节点，-2表示叶节点的特征阈值
            3、图上的数值是保留了3位小数的浮点数结果
            4、left数组是根节点（不含根节点本身）左子树前序遍历的节点id序列（含叶节点）
            5、right数组是根节点（不含根节点本身）的右子树前序遍历的节点id序列（含叶节点）
            6、threshold和features的顺序就是left——>right顺序
            7、feature：size类型，代表了当前节点用于分裂的特征索引，即在训练集中用第几列特征进行分裂
            8、n_node_samples：size类型，代表了训练时落入到该节点的样本总数。显然，父节点的n_node_samples将等于其左右子节点的n_node_samples之和
            这里应该提取满足的该变量预测种类的树的规则
            """
            rulesFromTree = extract_rules_from_tree(tree[0][0].tree_, count,feature_names=self.feature_names)
            self.rulesAll.update(rulesFromTree)
    def filter_rules(self, func):
        self.rules = filter(lambda x: func(x), self.rules)

    def filter_short_rules(self, k):
        self.filter_rules(lambda x: len(x.conditions) > k)

    def transform(self, X1,XOthers):
        """Transform dataset.
        Parameters
        ----------
        X:      array-like matrix, shape=(n_samples, n_features)
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, n_out)
            Transformed dataset. Each column represents one rule.
        """
        rule_list = list(self.rulesAll)
        # a = np.array([rule.transform(X) for rule in rule_list]).T
        # 这里 rule.transform调用的  Rule() 类中的transform
        # 每一条完整决策路径进行一次transform()
        # 找出满足一条完整决策路径的样本 值为 1 的就满足此决策路径
        # .T 后 得到行为样本 列为决策路径，每行对应的列为1的为该样本满足该列决策路径的规则（最终决策结果对不对另说)
        rule1 = np.array([rule.transform(X1) for rule in rule_list]).T
        ruleOthers = np.array([rule.transform(XOthers) for rule in rule_list]).T
        return rule1,ruleOthers

    # def __str__(self):
    #     return (map(lambda x: x.__str__(), self.rules)).__str__()
class Tree(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            tree_size=6,
            sample_fract='default',
            # max_rules=2000,
            max_rules=200,
            memory_par=0.1,
            rand_tree_size=True,  # 是否随机叶子结点大小
            random_state=None):
        self.rand_tree_size = rand_tree_size
        self.sample_fract = sample_fract
        self.max_rules = max_rules
        self.memory_par = memory_par
        self.tree_size = tree_size
        self.random_state = random_state

    def fit(self, X, y, feature_names=None):
        """Fit and estimate linear combination of rule ensemble
        """
        ## Enumerate features if feature names not provided
        N = X.shape[0]  # 数据个数
        if feature_names is None:
            self.feature_names = ['feature_' + str(x) for x in range(0, X.shape[1])]
        else:
            self.feature_names = feature_names  # 特征名称

        ## initialise tree generator
        # tree_generator = None 需要进行树初始化
        # tree_size:终端结点的平均数量
        # max_rules：后续拟合所用的规则数量
        n_estimators_default = int(np.ceil(self.max_rules / self.tree_size))
        self.sample_fract_ = min(0.5, (100 + 6 * np.sqrt(N)) / N) #N为数据个数
        # n_estimators: 弱学习器的数目
        # max_depth: 每一个学习器的最大深度，限制回归树的节点数目
        """
        subsample : float, optional (default=1.0)
        # 正则化中的子采样，防止过拟合，取值为(0,1]
        # 如果取值为1，则使用全部样本，等于没有使用子采样。
        # 如果取值小于1，则只使用一部分样本
        # 选择小于1的比例可以防止过拟合，但是太小又容易欠拟合。推荐在 [0.5, 0.8] 之间
        # 这里的子采样和随机森林的不一样，随机森林使用的是放回抽样，而这里是不放回抽样
        """
        self.tree_generator = GradientBoostingClassifier(n_estimators=n_estimators_default,
                                                         max_leaf_nodes=self.tree_size,
                                                         learning_rate=self.memory_par,
                                                         # subsample=self.sample_fract_,
                                                         random_state=self.random_state,max_depth=100)


        ## fit tree generator
        if not self.rand_tree_size:  # simply fit with constant tree size
            self.tree_generator.fit(X, y)
        else:  #  根据论文3.3随机化树的大小
            np.random.seed(self.random_state)
            tree_sizes = np.random.exponential(scale=self.tree_size - 2,
                                               size=int(np.ceil(self.max_rules*2 / self.tree_size)))
            tree_sizes = np.asarray([2 + np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))], dtype=int)
            i = int(len(tree_sizes) / 4)
            while np.sum(tree_sizes[0:i]) < self.max_rules:
                i = i + 1
            tree_sizes = tree_sizes[0:i]
            # 如果warm_start=True就表示就是在模型训练的过程中，在前一阶段的训练结果上继续训练
            self.tree_generator.set_params(warm_start=True)
            curr_est_ = 0
            for i_size in np.arange(len(tree_sizes)):
                size = tree_sizes[i_size]
                self.tree_generator.set_params(n_estimators=curr_est_ + 1)
                self.tree_generator.set_params(max_leaf_nodes=size)
                random_state_add = self.random_state if self.random_state else 0
                # warm_state=True 似乎重置了random_state，这样树是高度相关的，除非我们在这里手动更改random_state。
                self.tree_generator.set_params(
                    random_state=i_size + random_state_add)
                self.tree_generator.get_params()['n_estimators']
                self.tree_generator.fit(np.copy(X, order='C'), np.copy(y, order='C'))

                curr_est_ = curr_est_ + 1
                print("第%s/%s棵决策树训练完成!"%(curr_est_,len(tree_sizes)))
            self.tree_generator.set_params(warm_start=False)

        # tree_list = self.tree_generator.estimators_  # 访问一棵树
        tree_list = [[x] for x in self.tree_generator.estimators_]
        N = -1
        for dt in tree_list:
            N += 1
            dot_data = export_graphviz(dt[0][0], out_file=None,
                                            feature_names=feature_names,
                                            filled=True, rounded=True,
                                            class_names=[0,1])
            graph = pydotplus.graph_from_dot_data(dot_data)
            graph.write_png('../tree_pic/'+str(N)+"_DTtree.png")
        """
        这里为规则提取构造数据集：提取规则的数据集为训练集，且为正确预测的数据
        """
        predictTrain = self.tree_generator.predict(X).tolist() #计算训练集上的预测结果 用来提取相应的规则
        self.flag_1 = []
        self.flag_others = []
        id = -1
        for GT in y:
            #为预测正确的数据样本创建索引
            id += 1 #预测值的索引
            if predictTrain[id] == GT: #预测值正确
                if GT == 1:
                    self.flag_1.append(id)
                else:
                    self.flag_others.append(id)
        #用上一步创造的四个标签的索引构造预测正确四个标签的的数据样本
        self.predictTrue1 =  np.zeros([len(self.flag_1), 444], dtype=np.float32)
        self.predictTrueOthers =  np.zeros([len(self.flag_others), 444], dtype=np.float32)

        count = -1 #计数
        for train_id in self.flag_1:
            # 构造1标签样本
            count += 1
            self.predictTrue1[count,:] = X[train_id,:]

        count = -1 #计数
        for train_id in self.flag_others:
            # 构造0标签样本
            count += 1
            self.predictTrueOthers[count,:] = X[train_id,:]

        """
        这里为规则提取构造数据集：提取规则的数据集为训练集，且为正确预测的数据
        """
        ## extract rules
        self.rule_ensemble = RuleEnsemble(tree_list=tree_list,
                                          result=predictTrain,
                                          feature_names=self.feature_names)
        """
        ## concatenate original features and rules
        # rule_ensemble 调用的 RuleEnsemble()类中的 transform(X)
        # Xi_rules : 行为正确预测标签i的样本 列为决策路径，每行对应的列为1的为该样本满足该列决策路径的规则（最终决策结果对不对另说)
        """
        self.X1_rules,self.X_Others_rules= self.rule_ensemble.transform(self.predictTrue1,self.predictTrueOthers)
        return self


    def predict(self, X):
        """Predict outcome for X
        """
        return self.tree_generator.predict(X)
