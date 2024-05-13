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
# pkin(["ガチグマ(暁)","パオジアン","ハバタクカミ","ウーラオス(水)","キラフロル"],["カイリュー","AS","いかさまダイス"])
# pkin(["ブリジュラス","オーガポン(岩)","ハバタクカミ","ウーラオス(水)","イーユイ"],["カイリュー","AS","いかさまダイス"])
# pkin(["キョジオーン","オーガポン(水)","ハバタクカミ","パオジアン","イーユイ"],["カイリュー","HAS","クリアチャーム"])
# pkin(["ディンルー","サーフゴー","ハバタクカミ","パオジアン","ウーラオス(水)"],["カイリュー","HB","ゴツゴツメット"])
# pkin(["ガチグマ(暁)","モロバレル","ハバタクカミ","ウーラオス(水)","イーユイ"],["カイリュー","AS","いかさまダイス"])
# pkin(["タケルライコ","ミミッキュ","ハッサム","テツノツツミ","ランドロス"],["カイリュー","HAS","こだわりハチマキ"])

# print(df_in.iat[0,0])

# df_in.to_csv('pokemondata.csv')
# df_out.to_csv('kairyudata.csv')

#------------------------------------特徴量の作成------------------------------------
# df_in = pd.read_csv('pokemondata.csv')
# #一列めの0を削除
# df_in = df_in.drop(df_in.columns[[0]],axis=1)

# # ポケモンの名前を特徴量01に変換
# for row in range(len(df_in)):
#     for col in range(len(df_in.columns)):
#         #ポケモンの名前が既出ならそのカラムに，初なら新しくカラムを作成
#         if (np.any(df_toku.columns.values == df_in.iat[row,col])):
#             df_toku.at[row,df_in.iat[row,col]] = 1
#         else:
#             df_toku[df_in.iat[row,col]] = 0
#             df_toku.at[row,df_in.iat[row,col]] = 1

#NANを0で置換
# df_toku = df_toku.fillna(0)
# #csvで保存
# df_toku.to_csv('tokuchouryou/toku.csv')

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

    return df_ans_new

#------データから正解値を抽出して割り振る関数--------------
def ans_DF(df_series,ans_name):
    df_out = pd.read_csv('kairyudata.csv')
    #一列めの0を削除
    df_out = df_out.drop(df_out.columns[[0]],axis=1)
    df_ans = pd.Series()

    for row in range(len(df_out)):
        df_ans.at[row] = df_series[df_out.loc[row,ans_name]]

    return df_ans


#-----------正解値---------------------
# #求めたい正解値のSeriesを作成
# df_kata = ans_series("kata")
# df_mochimono = ans_series("mochimono")

# #正解値を作成
# df_ans_kata = ans_DF(df_kata,"kata")
# df_ans_kata = ans_DF(df_kata,"kata")

#-----------機械学習-----------------
#目的変数と説明変数のndaaryを作成
toku_nd = pd.read_csv("tokuchouryou/toku.csv").values[:,1:]
kata_nd = pd.read_csv("seikai/ans_kata.csv").values[:,1:]

print(kata_nd)

# #正規化
# scaler = StandardScaler()
# toku_nd = scaler.fit_transform(toku_nd)

X_train, X_test, y_train, y_test = train_test_split(toku_nd, kata_nd, test_size=0.2, random_state=42)
# print(X_train)
# print(y_train)
# print(X_test[:,0].size)

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
