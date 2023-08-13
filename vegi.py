# 基本ライブラリ
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# データセット読み込み
url = 'https://github.com/ozaku0928/vege_streamlit/blob/master/vege.csv?raw=true'
df = pd.read_csv(url)
#,index_col=0

# 目標値
# df['target'] = df.target

# 目標値を数字から野菜の名前に変更
df.loc[df['target'] == 0, 'target'] = '小松菜'
df.loc[df['target'] == 1, 'target'] = 'ほうれん草'

# 予測モデルの構築
x = df.drop('target',axis=1).values
y = df['target'].values

# ロジスティック回帰
#clf = LogisticRegression()
#clf.fit(x, y)


# モデルの定義
dtree = DecisionTreeClassifier(random_state=0)
#clf = LogisticRegression()
# モデルの学習
dtree.fit(x,y)

# サイドバーの入力画面
st.sidebar.header('Input features')

stemValue = st.sidebar.slider('stem length (cm)', min_value=0.2, max_value=2.0, step=0.1)
leafValue = st.sidebar.slider('leaf length (cm)', min_value=2.5, max_value=11.0, step=0.1)

# メインパネル
st.title('Vegetables Classifier')
st.write('## Input value')

# インプットデータ（1行のデータフレーム）
#value_df = pd.DataFrame({'stem length (cm)':stemValue, 'leaf length (cm)':leafValue}, index=[0])
#record = pd.Series([stemValue, leafValue], index=value_df.columns)
#value_df = value_df.append(record, ignore_index=True)

# インプットデータ（1行のデータフレーム）
#value_df = pd.DataFrame({'data':'data', 'stem length (cm)':stemValue, 'leaf length (cm)':leafValue}, index=[0])
#value_df.set_index('data', inplace=True)

value_df = pd.DataFrame([],columns=['data','stem length (cm)','leaf length (cm)'])
record = pd.Series(['data',stemValue, leafValue], index=value_df.columns)
value_df = value_df.append(record, ignore_index=True)
value_df.set_index("data",inplace=True)

# 入力値の値
st.write(value_df)

# 予測値のデータフレーム
pred_probs = dtree.predict_proba(value_df)
pred_df = pd.DataFrame(value_df,columns=['ほうれん草','小松菜'],index=['probability'])

#st.write('## Prediction')
#st.write(pred_df)

features = [[stemValue, leafValue]]
prediction = dtree.predict(features)

species = {
    0: "小松菜",
    1: "ほうれん草"
}

# 予測結果の出力
#def vegename(target):
#    if target == 0:
#        return '小松菜'
#    elif target == 1:
#        return 'ほうれん草'
name = prediction.tolist()
#.idxmax(axis=1)
st.write('## Vegetables')
st.write('この野菜はきっと',str(name[0]),'です!')
#st.write('この野菜はきっと',species[prediction[0]],'です!')