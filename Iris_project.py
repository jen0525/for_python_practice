#!/usr/bin/env python
# coding: utf-8

# # 6-1. 프로젝트 (1) load_digits : 손글씨를 분류해 봅시다

# (1) 필요한 모듈 import하기

# In[3]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
digits = load_digits() # Dictionary 자료형과 유사한 sklearn.utils.Bunch 자료형 
digits.keys()


# (2) 데이터 준비

# digits_data = digits.data

# (3) 데이터 이해하기

# In[8]:


import pandas as pd


# In[11]:


digits_df = pd.DataFrame(data= digits_data, columns =digits.feature_names)
digits_df


# In[13]:


digits_data.shape


# In[5]:


digits_label = digits.target # 각 이미지가 나타내는 숫자를 의미합니다.
print(digits_label.shape)
digits_label[:20]


# In[15]:





# In[ ]:


# 각 Label을 참조하여, 해당 이미지가 나타내는 숫자가 3이라면 3을 할당하고, 3이 아니라면 0을 할당하여 새로운 칼럼 new_label을 생성합니다.
new_label = [3 if i == 3 else 0 for i in digits_label] 
new_label[:20]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(digits_data,
                                                    new_label,
                                                    test_size=0.2,
                                                    random_state=13)
decision_tree = DecisionTreeClassifier(random_state=15)
decision_tree.fit(X_train, y_train) # 의사결정나무 모델로 학습
y_pred = decision_tree.predict(X_test) # 테스트 결과 예측

accuracy = accuracy_score(y_test, y_pred) # y_pred(답안지)를 y_test(정답지)로 채점
accuracy # 정확도 출력

