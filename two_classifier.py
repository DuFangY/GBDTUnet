import numpy as np
import pandas as pd
import pickle
import joblib
from Gbdt import Tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def fit_bg_onlyGDBT():
    """
    训练数据
    :return:
    """
    data = pd.read_csv("./two_classifier/bg.csv")
    data = data.fillna(0)
    y = data.zLabel.values
    X = data.drop("zLabel", axis=1)
    features = X.columns
    X = X.values
    X = X.astype(np.float32)
    X = StandardScaler().fit_transform(X)  #标准化
    X = pd.DataFrame(X)
    X = X.values

    gdbt_tree = Tree()
    gdbt_tree.fit(X, y, feature_names=features)
    path = './GDBT_only_bg.pkl'
    joblib.dump(gdbt_tree,path)

    path = './GDBT_only_bg.pkl'
    gdbt_tree = joblib.load(path)
    n_nodes_ = gdbt_tree.n_nodes_
    children_left_ = gdbt_tree.children_left_
    children_right_ = gdbt_tree.children_right_
    feature_ = gdbt_tree.feature_
    threshold_ = gdbt_tree.threshold_

    """
    测试数据
    """
    data = pd.read_csv("./two_classifier/bg_test.csv")
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
    pre = gdbt_tree.predict(X)
    accuracy = accuracy_score(y, pre)
    print("\n分类准确度为： ",accuracy)
    estimator = gdbt_tree.tree_generator
    # for i,e in enumerate(estimator.estimators_):
    #     print("_______________________________")
    #     print("Tree %d\n"%i)
    #     values = []
    #     V = explore_tree(estimator.estimators_[i][0],n_nodes_[i],children_left_[i],
    #              children_right_[i], feature_[i],threshold_[i],X,
    #              suffix=i, sample_id=9, feature_names=features)
    #     values.append(V)
    # print('\n'*2)
    # print(np.mean(values))
    # print(pre[9])

def fit_1_onlyGDBT():
    """
    训练数据
    :return:
    """
    data = pd.read_csv("./two_classifier/1.csv")
    data = data.fillna(0)
    y = data.zLabel.values
    X = data.drop("zLabel", axis=1)
    features = X.columns
    X = X.values
    X = X.astype(np.float32)
    X = StandardScaler().fit_transform(X)  #标准化
    X = pd.DataFrame(X)
    X = X.values

    gdbt_tree = Tree()
    gdbt_tree.fit(X, y, feature_names=features)
    path = './GDBT_only_1.pkl'
    joblib.dump(gdbt_tree,path)

    path = './GDBT_only_1.pkl'
    gdbt_tree = joblib.load(path)
    n_nodes_ = gdbt_tree.n_nodes_
    children_left_ = gdbt_tree.children_left_
    children_right_ = gdbt_tree.children_right_
    feature_ = gdbt_tree.feature_
    threshold_ = gdbt_tree.threshold_

    """
    测试数据
    """
    data = pd.read_csv("./two_classifier/1_test.csv")
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
    pre = gdbt_tree.predict(X)
    accuracy = accuracy_score(y, pre)
    print("\n分类准确度为： ",accuracy)

def explore_tree(estimator, n_nodes, children_left,children_right, feature,threshold,X_test,
                 suffix='', print_tree= False, sample_id=0, feature_names=None):

    if not feature_names:
        feature_names = feature


    # assert len(feature_names) == X.shape[1], "The feature names do not match the number of features."

    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes"
          % n_nodes)
    if print_tree:
        print("Tree structure: \n")
        for i in range(n_nodes):
            if is_leaves[i]:
                print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            else:
                print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                      "node %s."
                      % (node_depth[i] * "\t",
                         i,
                         children_left[i],
                         feature[i],
                         threshold[i],
                         children_right[i],
                         ))
            print("\n")
        print()

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    #sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    # print(X_test[sample_id,:])

    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        # tabulation = " "*node_depth[node_id] #-> makes tabulation of each level of the tree
        tabulation = ""
        if leave_id[sample_id] == node_id:
            print("%s==> Predicted leaf index \n"%(tabulation))
            #continue

        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("%sdecision id node %s : (X_test[%s, '%s'] (= %s) %s %s)"
              % (tabulation,
                 node_id,
                 sample_id,
                 feature_names[feature[node_id]],
                 X_test[sample_id, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))
    print("%sPrediction for sample %d: %s"%(tabulation,
                                            sample_id,
                                            estimator.predict(X_test)[sample_id]))
    return estimator.predict(X_test)[sample_id]


if __name__ == '__main__':
    fit_bg_onlyGDBT()
    # fit_1_onlyGDBT()