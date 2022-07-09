import pandas as pd
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

def XG_train(train,GT):
    print("XGboost开始训练！\n")
    XG = xgb.XGBClassifier(num_class=4,max_depth=3, learning_rate=0.1, n_estimators=36,verbosity=1,random_state=13, nthread=10,
                           eval_metric='mlogloss')
    XG.fit(train,GT)
    print("XGboost已经训练完成！\n")
    return XG

if __name__ == '__main__':
    data = pd.read_csv("slice_feature_all.csv")
    data = data.fillna(0)
    y = data.zLabel.values
    X = data.drop("zLabel", axis=1)
    features = X.columns
    X = X.values
    X = StandardScaler().fit_transform(X)  #标准化
    X = pd.DataFrame(X)

    XG = XG_train(X,y)
    # tree = XG.estimators_

    data = pd.read_csv("test.csv")
    data = data.fillna(0)
    y = data.zLabel.values
    X = data.drop("zLabel", axis=1)
    features = X.columns
    X = X.values
    X = StandardScaler().fit_transform(X)  #标准化
    X = pd.DataFrame(X)

    result = XG.predict(X)
    # contrast = dict(zip(['预测结果','真实结果'],[result,y]))  #生成字典以便生成表格
    # contrast = pd.DataFrame(contrast)
    # print(contrast)
    accuracy = accuracy_score(y, result)
    print("\n分类准确度为： ",accuracy)