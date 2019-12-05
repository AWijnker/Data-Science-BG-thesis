#!/usr/bin/env python
# coding: utf-8

# ## BoA stress prediction model

# In[1]:


#pip install catboost


# In[3]:


import pandas as pd
import re as re
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[4]:


from sklearn.metrics import confusion_matrix 
import numpy as np


# In[5]:


## loading data
boa_train_x = pd.read_csv("boa_dur_train_x.csv")
boa_train_y = pd.read_csv("boa_dur_train_y.csv")
boa_test_x = pd.read_csv("boa_dur_test_x.csv")
boa_test_y = pd.read_csv("boa_dur_test_y.csv")


# ## creating top 15 frequency plot

# In[8]:


boa_x = pd.concat([boa_train_x, boa_test_x])


# In[9]:


boa_sum = boa_x.sum()
boa_sum = pd.DataFrame(boa_sum)


# In[10]:


#boa_index = list(boa_sum.index)
#boa_sum.insert(0, "Categories", boa_index, True)


# In[11]:


boa_sum.rename(columns={ boa_sum.columns[0]: "Frequency" }, inplace = True)
boa_sum = boa_sum.sort_values(by = ['Frequency'], ascending = False)


# In[14]:


boa_top = boa_sum[:15]
boa_top = boa_top.reset_index()


# In[15]:


names = np.unique(boa_top['index'])


# In[16]:


index = ['Whatsapp_Messenger', 'Instagram', 'Phone_Tools', 'Google_Chrome', 'Snapchat', 
         'Instant_Messaging', 'Facebook', 'Social_Networking', 'Spotify', 'Camera', 'Youtube',
         'Internet_Browser', 'Ethica', 'Phone_Optimisation', 'Dialer']
boa_top['index'] = index


# In[ ]:


import matplotlib.pyplot as plt
fig, plot = plt.subplots()
plot.barh(boa_top['index'], boa_top['Frequency'])
plot.set_xlabel("Total number of occurrences")
plot.set_title('Top 15 most frequent features BoA model')
plt.gcf().subplots_adjust(left=0.30)
plot.set_ylim(len(names)-0.5, -0.5)
plt.savefig('frequency_plot_sequences_BoA.png')
plt.show()


# In[ ]:





# ## Using XGClassifier

# In[ ]:





# In[ ]:





# In[4]:


model = svm.SVC(random_state= 2019, C = 10, kernel = 'rbf')


# In[12]:


#tuning


# In[ ]:


## tuning
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'kernel': ['rbf']},
]
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
clf.fit(boa_train_x, boa_train_y)  
print('Best score for data1:', clf.best_score_) 
print('Best C:',clf.best_estimator_.C) 
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)


# In[ ]:





# In[ ]:


model = svm.SVC(random_state= 2019, C = 10, kernel = 'rbf', gamma = 0.001)
model.fit(x_ada, y_ada)
#y_pred = model.predict(X_test)


# In[10]:


y_pred = model.predict(boa_test_x)


# In[ ]:


accuracy = accuracy_score(boa_test_y, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#(boa without categories)


# In[ ]:





# In[ ]:


## XGB


# In[13]:


model = XGBClassifier(random_state = 2019, max_depth = 2, n_estimators = 100)


# In[ ]:


## tuning 
optimization_dict = {'max_depth': [2,4,6],
                     'n_estimators': [50,100,150]}
model_XGB = GridSearchCV(model, optimization_dict, 
                     scoring='accuracy', verbose=1)
model_XGB.fit(boa_train_x,boa_train_y)
print(model_XGB.best_score_)
print(model_XGB.best_params_)


# In[ ]:


model.fit(x_ada, y_ada)


# In[ ]:


boa_test_x = boa_test_x.as_matrix()
y_pred = model.predict(boa_test_x)
accuracy = accuracy_score(boa_test_y, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100))


# ## confusion matrix for BoA

#  evaluation set possible

# In[29]:


import numpy as np
y_pred_df = pd.DataFrame(y_pred)
y_pred_df[0].value_counts()
labels = np.unique(boa_test_y)


# In[ ]:





# In[25]:


import matplotlib.pyplot as mplt
import scikitplot as skp


# In[76]:


BoA_cm = confusion_matrix(boa_test_y, y_pred)


# In[ ]:


BoA_cm = skp.metrics.plot_confusion_matrix(boa_test_y, y_pred, normalize=True, title = 'Normalised Confusion Matrix BoA model')
BoA_cm.set_ylim(len(labels)-0.5, -0.5)
mplt.savefig('CM for BoA model.png')
mplt.show()


# In[14]:


from sklearn.metrics import recall_score


# In[ ]:





# In[ ]:


recall = recall_score(boa_test_y, y_pred, average = 'macro')
recall


# ## ADASYN with BoA model

# In[12]:


from imblearn.over_sampling import ADASYN
import numpy as np


# In[ ]:


adasyn_model = ADASYN(sampling_strategy = 'minority', random_state = 2019, n_neighbors = 5)
x_ada, y_ada = adasyn_model.fit_resample(boa_train_x, boa_train_y)


# In[15]:


y_ada = pd.DataFrame(y_ada)
y_ada[0].value_counts()


# In[ ]:


model = XGBClassifier(random_state = 2019, max_depth = 2, n_estimators = 100)
kfold = KFold(n_splits=10, random_state=2019)
results = cross_val_score(model, boa_train_x, boa_train_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*200))


# In[22]:


model = XGBClassifier(random_state = 2019, max_depth = 2, n_estimators = 100)
model.fit(x_ada, y_ada)
#boa_test_x = boa_test_x.as_matrix()
y_pred = model.predict(boa_test_x)


# In[23]:


labels = np.unique(boa_test_y)


# In[ ]:





# In[31]:


y_TEST = pd.DataFrame(boa_train_y)
y_TEST['stressed'].value_counts()


# In[26]:


BoA_cm = confusion_matrix(boa_test_y, y_pred)
BoA_cm = skp.metrics.plot_confusion_matrix(boa_test_y, y_pred, normalize=True, title = 'Normalised Confusion Matrix BoA model with binarised stress levels')
BoA_cm.set_ylim(len(labels)-0.5, -0.5)
mplt.savefig('CM for BoA model BINARISED.png')
mplt.show()


# In[ ]:





# ## Feature importance

# In[32]:


from xgboost import plot_importance


# In[37]:


model = XGBClassifier(random_state = 2019, max_depth = 2, n_estimators = 100)


# In[61]:


columns = list(boa_train_x.columns) 


# In[63]:


x_ada_df = pd.DataFrame(x_ada, columns = columns)


# In[64]:


XGB_model = model.fit(x_ada_df, y_ada)


# In[103]:


plot_importance(model, max_num_features = 15, title =  'Feature Importance BoA model')
mplt.savefig('Feature Importance BoA model.png', bbox_inches='tight')
mplt.show()


# In[ ]:





# ## Binarised stress

# In[4]:


## loading data
boa_train_x = pd.read_csv("boa_dur_train_x.csv")
boa_train_y = pd.read_csv("boa_dur_train_y.csv")
boa_test_x = pd.read_csv("boa_dur_test_x.csv")
boa_test_y = pd.read_csv("boa_dur_test_y.csv")


# In[5]:


boa_train_y.loc[boa_train_y.stressed < 2, 'stressed'] = 0
boa_test_y.loc[boa_test_y.stressed < 2, 'stressed'] = 0


# In[6]:


boa_train_y.loc[boa_train_y.stressed > 1, 'stressed'] = 1
boa_test_y.loc[boa_test_y.stressed > 1, 'stressed'] = 1


# In[ ]:





# In[7]:


## xgb


# In[8]:


# train


# In[9]:


model = XGBClassifier(random_state = 2019, max_depth = 2, n_estimators = 100)
kfold = KFold(n_splits=10, random_state=2019)
results = cross_val_score(model, boa_train_x, boa_train_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[78]:


#test


# In[79]:


model = XGBClassifier(random_state = 2019, max_depth = 2, n_estimators = 100)
model.fit(boa_train_x, boa_train_y)


# In[80]:


y_pred = model.predict(boa_test_x)
accuracy = accuracy_score(boa_test_y, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100))


# In[ ]:





# In[82]:


## svm


# In[83]:


##train


# In[10]:


model = svm.SVC(random_state= 2019, C = 10, kernel = 'rbf', gamma = 0.001)
kfold = KFold(n_splits=10, random_state=2019)
results = cross_val_score(model, boa_train_x, boa_train_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[84]:


## test


# In[85]:


model = svm.SVC(random_state= 2019, C = 10, kernel = 'rbf', gamma = 0.001)
model.fit(boa_train_x, boa_train_y)


# In[86]:


y_pred = model.predict(boa_test_x)
accuracy = accuracy_score(boa_test_y, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100))


# In[ ]:





# ## Feature importance binarised stress

# In[90]:


from xgboost import plot_importance


# In[105]:


model = XGBClassifier(random_state = 2019, max_depth = 2, n_estimators = 100)


# In[106]:


columns = list(boa_train_x.columns) 


# In[107]:


XGB_model = model.fit(boa_train_x, boa_train_y)


# In[ ]:





# In[110]:


import matplotlib.pyplot as mplt
plot_importance(model, max_num_features = 15, title =  'Top 15 feature importance BoA model with binarised stress levels')
mplt.savefig('Feature Importance BoA model Binarised.png', bbox_inches='tight')
mplt.show()


# In[ ]:




