#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import re as re
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[49]:


from imblearn.over_sampling import ADASYN


# In[50]:


spm_daily_final08 = pd.read_csv("spm_daily_final08_4.csv")


# In[51]:


spm_daily_final08 = spm_daily_final08.iloc[:,5:]


# In[52]:


x08 = spm_daily_final08.iloc[:, :-1]
y08 = spm_daily_final08[spm_daily_final08.columns[-1]]


# In[53]:


xTrain08, xTest08, yTrain08, yTest08 = train_test_split(x08, y08, test_size = 0.3, random_state = 0)


# In[54]:


##adasyn


# In[ ]:





# In[55]:


adasyn_model = ADASYN(sampling_strategy = 'minority', random_state = 2019, n_neighbors = 5)


# In[56]:


x_ada, y_ada = adasyn_model.fit_resample(xTrain08, yTrain08)


# In[ ]:





# In[57]:


##SMOTE


# In[58]:


from imblearn.over_sampling import SMOTE


# In[59]:


x_smote, y_smote = SMOTE().fit_sample(xTrain08, yTrain08)


# In[60]:


#testing adasyn


# In[ ]:


y_smote = pd.DataFrame(y_smote)


# In[ ]:





# In[ ]:





# In[95]:


## svm as classifier


# In[44]:


model = svm.SVC(random_state= 2019, C = 10, kernel = 'rbf')


# In[ ]:


model.fit(x_ada, y_ada)


# In[ ]:


y_pred = model.predict(xTest08)
y_pred_df = pd.DataFrame(y_pred)
y_pred_df[0].value_counts()


# In[ ]:


accuracy = accuracy_score(yTest08, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100))


# In[50]:


model = XGBClassifier(max_depth = 2, n_estimators = 100)


# In[ ]:


model.fit(xTrain08, yTrain08, verbose = True)


# In[ ]:


y_pred = model.predict(xTest08)


# In[ ]:





# In[ ]:


accuracy = accuracy_score(yTest08, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100))


# In[ ]:


model = XGBClassifier()


# In[ ]:


kfold = KFold(n_splits=10, random_state=2019)
results = cross_val_score(model, xTrain08, yTrain08, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

