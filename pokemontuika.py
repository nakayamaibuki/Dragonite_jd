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
df_toku = pd.DataFrame()
df_ans = pd.DataFrame()


#------------------新しいデータをデータフレームに格納------------
def pkin(Data_in,Data_out):
    global df_in,df_out,colums_in,colums_out
    df_in_new = pd.Series(data=Data_in,index=colums_in)
    df_out_new = pd.Series(data=Data_out,index=colums_out)
    # print(df_in_new)
    df_in = pd.concat([df_in,df_in_new.to_frame().T])
    df_out = pd.concat([df_out,df_out_new.to_frame().T])
    # print(df_out)

#-----------------------新しいデータの入力--------------------------------------
pkin(["ガチグマ(暁)","パオジアン","ハバタクカミ","ウーラオス(水)","キラフロル"],["カイリュー","AS","いかさまダイス"])
pkin(["ブリジュラス","オーガポン(岩)","ハバタクカミ","ウーラオス(水)","イーユイ"],["カイリュー","AS","いかさまダイス"])
pkin(["キョジオーン","オーガポン(水)","ハバタクカミ","パオジアン","イーユイ"],["カイリュー","HAS","クリアチャーム"])
pkin(["ディンルー","サーフゴー","ハバタクカミ","パオジアン","ウーラオス(水)"],["カイリュー","HB","ゴツゴツメット"])
pkin(["ガチグマ(暁)","モロバレル","ハバタクカミ","ウーラオス(水)","イーユイ"],["カイリュー","AS","いかさまダイス"])
pkin(["タケルライコ","ミミッキュ","ハッサム","テツノツツミ","ランドロス"],["カイリュー","HAS","こだわりハチマキ"])

df_in.to_csv('pokemondata.csv')
df_out.to_csv('kairyudata.csv')