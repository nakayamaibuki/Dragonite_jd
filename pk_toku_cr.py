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

# ------------------------------------特徴量の作成------------------------------------
df_in = pd.read_csv('learn_data/pk_data.csv')
#一列めの0を削除
df_in = df_in.drop(df_in.columns[[0]],axis=1)

# ポケモンの名前を特徴量01に変換
for row in range(len(df_in)):
    for col in range(len(df_in.columns)):
        #ポケモンの名前が既出ならそのカラムに，初なら新しくカラムを作成
        if (np.any(df_toku.columns.values == df_in.iat[row,col])):
            df_toku.at[row,df_in.iat[row,col]] = 1
        else:
            df_toku[df_in.iat[row,col]] = 0
            df_toku.at[row,df_in.iat[row,col]] = 1

# NANを0で置換
df_toku = df_toku.fillna(0)
#csvで保存
df_toku.to_csv('tokuchouryou/toku.csv')



