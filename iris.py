# streamlit은 ipynb와 연동이 안 되어서 .py로 만들어서 함 
# 기본적으로 비동기처리를 지원하지 않음
# 모바일 디바이스에서는 제한적
# ctrl+shift+p --> jupyter:대화형창만들기 클릭
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Species Predictor")
st.write("""This app predicts the Iris Flower species""")

iris= load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]   #target의 값을 ['species'로 추가]
#print(df.head())
# 0: setosa  1:versicolor  2:virginica

#사이드바에서 입력받기
st.sidebar.header("Input Parameters")
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal length(cm)", 
                                    float(df['sepal length (cm)'].min()),
                                    float(df['sepal length (cm)'].max()),
                                    float(df['sepal length (cm)'].mean())
    )
    sepal_width = st.sidebar.slider("Sepal width(cm)", 
                                    float(df['sepal width (cm)'].min()),
                                    float(df['sepal width (cm)'].max()),
                                    float(df['sepal width (cm)'].mean())
    )
    petal_length = st.sidebar.slider("Petal length(cm)", 
                                    float(df['petal length (cm)'].min()),
                                    float(df['petal length (cm)'].max()),
                                    float(df['petal length (cm)'].mean())
    )
    petal_width = st.sidebar.slider("Petal width(cm)", 
                                    float(df['petal width (cm)'].min()),
                                    float(df['petal width (cm)'].max()),
                                    float(df['petal width (cm)'].mean())
    )
    data = {'sepal length(cm)':sepal_length,
            'sepal width(cm)':sepal_width,
            'petal length(cm)':petal_length,
            'petal width(cm)':petal_width,
            }
    features=pd.DataFrame(data, index=[0])     
    return features   

input_df = user_input_features()    

# 사용자 입력값 표시
st.subheader("User Input Parameters")
st.write(input_df)

#모델학습하기
model=RandomForestClassifier(random_state=42)
model.fit(X, y)

#예측 데이터프레임 형식을 넘파이형식으로 변환하여 입력
prediction = model.predict(input_df.to_numpy())
prediction_proba = model.predict_proba(input_df.to_numpy())

st.subheader("Prediction")
st.write(prediction)

st.subheader("Prediction Probability")
st.write(prediction_proba)

#histogram
st.subheader("Histogram of Features")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    sns.histplot(df[iris.feature_names[i]], kde=True, ax=ax)
    ax.set_title(iris.feature_names[i])
plt.tight_layout()
st.pyplot(fig)    

#correlation
st.subheader("Correlation Matrix")
numerical_df = df.drop('species', axis=1)
corr_matrix=numerical_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
plt.tight_layout()
st.pyplot(fig)    

#pairplot
st.subheader("PairPlot")
fig = sns.pairplot(df, hue="species")
plt.tight_layout()
st.pyplot(fig) 

#피처 중요도 시각화
st.subheader("Feature Importance")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  #내림차순으로

plt.figure(figsize=(10, 4))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [iris.feature_names[i] for i in indices])
plt.xlim([-1, X.shape[1]])
st.pyplot(plt)