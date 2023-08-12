# 基本ライブラリ
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# データセット読み込み
df = pd.read_csv('/Users/ozaku/Desktop/vegitable/vege.csv')

# 目標値
df['target'] = df.target

# 目標値を数字から野菜の名前に変更
df.loc[df['target'] == 0, 'target'] = '小松菜'
df.loc[df['target'] == 1, 'target'] = 'ほうれん草'

# 予測モデルの構築
x = df.drop('target', axis=1).values
y = df['target'].values

# モデルの定義
dtree = DecisionTreeClassifier(random_state=0)

# モデルの学習
dtree.fit(x, y)

# サイドバーの入力画面
st.sidebar.header('Input features')

stemValue = st.sidebar.slider('stem length (cm)', min_value=0.2, max_value=2.0, step=0.1)
leafValue = st.sidebar.slider('leaf length (cm)', min_value=2.5, max_value=11.0, step=0.1)

# メインパネル
st.title('Vegetables Classifier')
st.write('## Input value')

# インプットデータ（1行のデータフレーム）
value_df = pd.DataFrame([],columns=['data','stem length (cm)','leaf length (cm)'])
record = pd.Series(['data',stemValue, leafValue], index=value_df.columns)
value_df = value_df.append(record, ignore_index=True)
value_df.set_index('data',inplace=True)


# 入力値の値
st.write(value_df)

# 予測値のデータフレーム
pred_probs = dtree.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs,columns=['ほうれん草','小松菜'],index=['probability'])

# 予測結果の出力
name = pred_df.idxmax(axis=1).tolist()
st.write('## Vegetables')
st.write('この野菜はきっと',str(name[0]),'です!')