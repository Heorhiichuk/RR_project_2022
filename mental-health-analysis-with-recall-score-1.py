#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
os.getcwd()


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


df=pd.read_excel('Cleaned Data.xlsx')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


cor=df.corr()


# In[8]:


cor


# In[9]:


fig = plt.figure(figsize=(22,20))

fig.add_axes([0,0,1,1])
ax = fig.get_axes()[0]
sns.heatmap(cor, ax=ax, vmin=-1, vmax=1, annot=True)


# In[10]:


cor.iloc[1]


# In[11]:


columns_to_drop=['Region','I have my regular access to the internet','I am currently employed at least part-time','I am on section 8 housing','I receive food stamps','Annual income from social welfare programs','I have a gap in my resume','Total length of any gaps in my resume in\xa0months.','Household Income','Device Type']


# In[12]:


df.drop(columns=columns_to_drop,inplace=True)


# In[13]:


df.shape


# In[14]:


df['Annual income (including any social welfare programs) in Rupee']=df['Annual income (including any social welfare programs) in USD']*70


# In[15]:


df.drop('Annual income (including any social welfare programs) in USD',axis=1,inplace=True)


# In[16]:


df.head(20)


# In[17]:


df.info()


# In[18]:


for i in df:
    if i=='Education' or i=='Age' or i=='Gender':
        df[i].dropna()
    else:
        df[i].fillna(df[i].median(),inplace=True)


# In[19]:


df.info()


# In[20]:


y=df['I identify as having a mental illness']


# In[21]:


df.drop('I identify as having a mental illness',axis=1,inplace=True)


# In[22]:


df.head()


# ## Label encoding

# In[23]:


from sklearn.preprocessing import LabelEncoder


# In[24]:


le_educatio=LabelEncoder()
le_age=LabelEncoder()
le_gender=LabelEncoder()


# In[25]:


df['Education']=le_educatio.fit_transform(df['Education'])


# In[26]:


df['Age']=le_age.fit_transform(df['Age'])
df['Gender']=le_gender.fit_transform(df['Gender'])


# In[27]:


df.head()


# ## standardizing data

# In[28]:


from sklearn.preprocessing import StandardScaler


# In[29]:


ss=StandardScaler()


# In[30]:


df2=ss.fit_transform(df)


# In[31]:


df2.shape


# In[32]:


X=df.values


# In[33]:


X.shape


# In[ ]:





# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[36]:


print(X_train.shape,y_train.shape)


# ## Training the Logistic regression model

# In[37]:


from sklearn.linear_model import LogisticRegression


# In[38]:


lr=LogisticRegression()


# In[39]:


lr.fit(X_train,y_train)


# In[40]:


y_pred=lr.predict(X_test)


# In[41]:


y_pred


# In[42]:


from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score


# In[43]:


accuracy_score(y_test,y_pred)


# In[44]:


cf_matrix=confusion_matrix(y_test,y_pred)


# In[45]:


sns.heatmap(cf_matrix, annot=True)


# In[46]:


recall_score(y_test,y_pred)


# In[47]:


precision_score(y_test,y_pred)


# # In this model we have accuracy of 86.56% on test data and recall score of 0.875 and precision rate of 0.66.
# # we will prefer going for more recall value as we want less False Negative

# In[48]:


from sklearn.ensemble import RandomForestClassifier


# In[49]:


rf=RandomForestClassifier()


# In[50]:


rf.fit(X_train,y_train)


# In[51]:


y_pred2=rf.predict(X_test)


# In[52]:


accuracy_score(y_test,y_pred2)


# In[53]:


recall_score(y_test,y_pred2)


# In[54]:


precision_score(y_test,y_pred2)


# In[55]:


cf_matrix2=confusion_matrix(y_test,y_pred2)


# In[56]:


sns.heatmap(cf_matrix2, annot=True)


# In[ ]:




