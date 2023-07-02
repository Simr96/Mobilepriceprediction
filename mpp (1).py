#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[34]:


train = pd.read_csv(r'C:\Users\User\Downloads\classical_ml_data\train.csv')
test = pd.read_csv(r'C:\Users\User\Downloads\classical_ml_data\test.csv')


# In[35]:


train.head()


# In[36]:


test.head()


# In[37]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_column', None)


# In[38]:


test.drop('id', axis = 1, inplace = True)


# In[39]:


test.head()


# In[40]:


sns.countplot(train['price_range'])


# In[41]:


train.shape, test.shape


# In[42]:


train.isnull().sum()


# In[43]:


train.info()


# In[44]:


test.info()


# In[45]:


train.describe()


# In[46]:


train.plot(x= 'price_range', y = 'ram', kind = 'scatter')
plt.show()


# In[47]:


train.plot(x= 'price_range', y = 'battery_power', kind = 'scatter')
plt.show()


# In[76]:


train.plot(x= 'price_range', y = 'int_memory', kind = 'scatter')
plt.show()


# In[77]:


train.plot(x= 'price_range', y = 'm_dep', kind = 'scatter')
plt.show()


# In[78]:


train.plot(x= 'price_range', y = 'dual_sim', kind = 'scatter')
plt.show()


# In[48]:


train.plot(x= 'price_range', y = 'n_cores', kind = 'scatter')
plt.show()


# In[49]:


plt.figure(figsize=(20,20))
sns.heatmap(train.corr(), annot= True, cmap=plt.cm.Accent_r)
plt.show()


# In[50]:


train.plot(kind = 'box', figsize = (20,10))


# In[56]:


X = train.drop('price_range', axis = 1)
y = train['price_range']


# In[57]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.1,random_state=101)


# In[58]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
test = sc.transform(test)


# In[59]:


X_train


# In[63]:


X_test


# In[64]:


test


# In[60]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train , Y_train)


# In[61]:


pred = dtc.predict(X_test)
pred


# In[62]:


from sklearn.metrics import accuracy_score, confusion_matrix
dtc_acc = accuracy_score(pred,Y_test)
print(dtc_acc)
print(confusion_matrix(pred,Y_test))


# In[65]:


from sklearn.svm import SVC
knn=SVC()
knn.fit(X_train,Y_train)


# In[66]:


pred1 = knn.predict(X_test)
pred1


# In[69]:


from sklearn.linear_model import LogisticRegression 

classifier= LogisticRegression(random_state=101)  
classifier.fit(X_train, Y_train)


# In[70]:


from sklearn.metrics import accuracy_score
svc_acc = accuracy_score(pred1,Y_test)
print(svc_acc)
print(confusion_matrix(pred1,Y_test))


# In[71]:


from sklearn.linear_model import LogisticRegression  
lr=LogisticRegression()


# In[72]:


lr.fit(X_train,Y_train)
LogisticRegression()
pred2 = lr.predict(X_test)
pred2


# In[73]:


lr_acc = accuracy_score(pred2,Y_test)
print(lr_acc)
print(confusion_matrix(pred2,Y_test))


# In[74]:


plt.bar(x=['dtc','svc','lr'],height=[dtc_acc,svc_acc,lr_acc])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.show()


# In[75]:


lr.predict(test)


# In[ ]:




