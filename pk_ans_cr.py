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

#--------------------変数------------------------------
colums_in = ["pk_1","pk_2","pk_3","pk_4","pk_5"]
colums_out = ["name","kata","mochimono"]
df_in = pd.DataFrame(columns=colums_in)
df_out = pd.DataFrame(columns=colums_out)
# print(df_in)

#----------正解データのSeries作成--------------
def ans_series(ans_name):
    df_out = pd.read_csv('kairyudata.csv')
    #一列めの0を削除
    df_out = df_out.drop(df_out.columns[[0]],axis=1)
    df_ans_new = pd.Series()

    for row in range(len(df_out)):
        if (np.any(df_ans_new.index.values == df_out.loc[row,ans_name])):
            pass
        else:
            if (len(df_ans_new.index) == 0):
                df_ans_new[df_out.loc[row,ans_name]] = 0
            else:
                df_ans_new[df_out.loc[row,ans_name]] = len(df_ans_new.index)

    df_ans_new.to_csv('seikai/label_'+ans_name+'.csv')

    return df_ans_new

#------データから正解値を抽出して割り振る関数--------------
def ans_DF(df_series,ans_name):
    df_out = pd.read_csv('kairyudata.csv')
    #一列めの0を削除
    df_out = df_out.drop(df_out.columns[[0]],axis=1)
    df_ans = pd.Series()

    for row in range(len(df_out)):
        df_ans.at[row] = df_series[df_out.loc[row,ans_name]]

    df_ans.to_csv('seikai/ans_'+ans_name+'.csv')

    return df_ans


#-----------正解値---------------------
#求めたい正解値のSeriesを作成
df_kata = ans_series("kata")
df_mochimono = ans_series("mochimono")

#正解値を作成
df_ans_kata = ans_DF(df_kata,"kata")
df_ans_kata = ans_DF(df_kata,"kata")
df_ans = pd.DataFrame()

