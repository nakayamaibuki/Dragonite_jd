import pandas as pd
import numpy as np
import pandas as pd

#--------------------変数------------------------------

#----------正解データのSeries作成--------------
def ans_series(ans_name):
    df_out = pd.read_csv('learn_data/Drago_data.csv')
    # #一列めの0を削除
    # df_out = df_out.drop(df_out.columns[[0]],axis=1)
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
def ans_label(df_series,ans_name):
    df_out = pd.read_csv('learn_data/Drago_data.csv')
    #一列めの0を削除
    df_out = df_out.drop(df_out.columns[[0]],axis=1)
    df_ans = pd.Series()

    for row in range(len(df_out)):
        df_ans.at[row] = df_series[df_out.loc[row,ans_name]]

    df_ans.to_csv('seikai/ans_'+ans_name+'.csv')

    return df_ans


#-----------正解値---------------------
#求めたい正解値のラベル表を作成
df_kata = ans_series("kata")
df_mochimono = ans_series("mochimono")

#正解値のラベルを作成を作成
df_ans_kata = ans_label(df_kata,"kata")
df_ans_kata = ans_label(df_mochimono,"mochimono")
df_ans = pd.DataFrame()

