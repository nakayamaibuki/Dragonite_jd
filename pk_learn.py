import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from scipy.stats import sem
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import MySQLdb

#--------------------変数------------------------------
colums_in = ["pk_1","pk_2","pk_3","pk_4","pk_5"]
colums_out = ["name","kata","mochimono"]
df_in = pd.DataFrame(columns=colums_in)
df_out = pd.DataFrame(columns=colums_out)
# print(df_in)
df_toku = pd.DataFrame()
df_ans = pd.DataFrame()

#-----------機械学習-----------------
def RF(ans_name):
    #目的変数と説明変数のndaaryを作成
    toku_nd      = pd.read_csv("tokuchouryou/toku.csv").values[:,1:]
    ans_nd      = pd.read_csv("seikai/ans_"+ans_name+".csv").values[:,1:]

    X_train, X_test, y_train, y_test = train_test_split(toku_nd, ans_nd, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators = 10,criterion="gini", max_depth=10,random_state=42)
    clf.fit(X_train,y_train)
    # 学習させたモデルを使ってテストデータに対する予測を出力する
    count = 0
    pred = clf.predict(X_test)
    for i in range(X_test[:,0].size):
        print('[{0}] correct:{1}, predict:{2}'.format(i, y_test[i], pred[i]))
        if pred[i] == y_test[i]:
                count += 1

    # # 予測結果から正答率を算出する
    # score = float(count) / test_size
    # print('{0} / {1} = {2}'.format(count, test_size, score))

    #-------交差検証------------------------
    # kf = StratifiedKFold(n_splits=6, shuffle=False)
    # for train_index, test_index in kf.split(toku_nd, kata_nd):

    #     clf = RandomForestClassifier(n_estimators = 10,criterion="entropy", max_depth=10)
    #     clf.fit(toku_nd[train_index],kata_nd[train_index])
    #     pred_test = clf.predict(toku_nd[test_index])
    #     f1score_test = f1_score(kata_nd[test_index], pred_test)
    #     cm = confusion_matrix(kata_nd[test_index], pred_test)

    #     pred_test = clf.predict(toku_nd[test_index])
    #     f1score_test = f1_score(kata_nd[test_index], pred_test)
    #     cm = confusion_matrix(kata_nd[test_index], pred_test)

    #     if (f1score_test>f1score):
    #         with open('model/model_kNN.pickle', mode='wb') as f:
    #             pickle.dump(clf,f,protocol=2)
    #     f1score = f1score_test
    #     ave = ave + cm

    # print(ave)

#------------実行--------------------
RF("kata")