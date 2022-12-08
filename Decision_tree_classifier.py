#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Social_Network_Ads.csv")
df.head()


# In[3]:


df.isna().sum()


# In[5]:


x=df.iloc[:,[2,3]]
y=df.iloc[:,-1]
y.head(2)


# In[6]:


sns.countplot(y)


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[10]:


x_train.shape


# In[11]:


x_test.shape


# In[12]:


x.shape


# In[13]:


y_train.shape


# In[15]:


y_test.shape


# In[17]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[18]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(x_train,y_train)


# In[26]:


y_pred=classifier.predict(x_test)
y_pred


# In[27]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)


# In[28]:


accuracy


# In[ ]:




