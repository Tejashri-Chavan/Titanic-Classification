#!/usr/bin/env python
# coding: utf-8

# In[153]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[154]:


train=pd.read_csv('train.csv')


# In[155]:


train.head()


# In[156]:


train.isnull()


# In[157]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[158]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[159]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[160]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[161]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[162]:


train['Age'].hist(bins=30,color='darkred',alpha=0.3)


# In[163]:


sns.countplot(x='SibSp',data=train)


# In[164]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[165]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[166]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
        
    else:
        return Age


# In[167]:


train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)


# In[168]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[169]:


train.drop('Cabin',axis=1,inplace=True)


# In[170]:


train.head()


# In[171]:


train.dropna(inplace=True)


# In[172]:


train.info()


# In[173]:


pd.get_dummies(train['Embarked'],drop_first=True).head()


# In[174]:


sex = pd.get_dummies(train['Sex'], drop_first=True)

embark=pd.get_dummies(train['Embarked'],drop_first=True)


# In[175]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[176]:


train.head()


# In[177]:


train=pd.concat([train,sex,embark],axis=1)


# In[178]:


train.head()


# In[179]:


train.drop('Survived',axis=1).head()


# In[180]:


train['Survived'].head()


# In[181]:


from sklearn.model_selection import train_test_split


# In[182]:


X_train,X_test,y_train,y_test=train_test_split(train.drop('Survived',axis=1),
                                              train['Survived'],test_size=0.30,
                                              random_state=101)


# In[183]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train_scaled, y_train)


# In[184]:


X_test = X_test.fillna(X_train.mean())


# In[185]:


predictions = logmodel.predict(X_test)


# In[186]:


from sklearn.metrics import confusion_matrix


# In[187]:


accuracy=confusion_matrix(y_test,predictions)


# In[188]:


accuracy


# In[189]:


from sklearn.metrics import accuracy_score


# In[190]:


accuracy=accuracy_score(y_test,predictions)
accuracy


# In[191]:


predictions


# In[192]:


from sklearn.metrics import classification_report


# In[193]:


print(classification_report(y_test,predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




