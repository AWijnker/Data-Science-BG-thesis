#!/usr/bin/env python
# coding: utf-8

# ## SPM_model_binary_and_categorical

# In[ ]:





# In[1]:


import pandas as pd
import re as re
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[2]:


from sklearn.metrics import confusion_matrix 


# #### Load mood

# In[3]:


mood_averages = pd.read_csv("mood_averages.csv")


# In[4]:


mood_averages["response_date"] = mood_averages["response_date"].apply(str)
mood_averages["user_id"] = mood_averages["user_id"].apply(str)


# In[ ]:





# #### Create final from combined

# In[ ]:


spm_daily_combined = pd.read_csv("spm_daily_combined_0.8_6.csv")


# In[8]:


rules = spm_daily_combined.iloc[0]


# In[9]:


rules = rules[4:]


# In[ ]:





# In[ ]:


for i in rules:
  spm_daily_combined[i] = spm_daily_combined["apps"].str.count(i, re.I)


# In[ ]:


spm_daily_combined


# In[104]:


spm_daily_combined["response_date"] = spm_daily_combined["response_date"].apply(str)


# In[105]:


spm_daily_combined["user_id"] = spm_daily_combined["user_id"].apply(str)


# In[106]:


## merge stress level onto dataset
spm_daily_final = pd.merge(spm_daily_combined, mood_averages, on = ("user_id", "response_date"))


# In[107]:


spm_daily_final = spm_daily_final.iloc[:,4:]


# In[ ]:





# In[109]:


spm_daily_final.to_csv("spm_daily_final08_5.csv")


# #### Final to model

# In[3]:


spm_daily_final = pd.read_csv("spm_daily_final08_6.csv")


# In[4]:


## making them binary
#spm_daily_final.loc[spm_daily_final.stressed < 2, 'stressed'] = 0


# In[5]:


#spm_daily_final.loc[spm_daily_final.stressed > 1, 'stressed'] = 1


# In[6]:


spm_daily_final = spm_daily_final.iloc[:,5:]


# In[ ]:





# In[7]:


x = spm_daily_final.iloc[:, :-1]
y = spm_daily_final[spm_daily_final.columns[-1]]


# In[ ]:





# In[9]:


##creating top 15 frequency column


# In[10]:


freq_plot = x.sum()
freq_plot = pd.DataFrame(freq_plot)


# In[ ]:





# In[ ]:





# In[216]:


#freq_plot.insert(0, "Sequence", freq_index, True) 


# In[11]:


freq_plot.rename(columns={ freq_plot.columns[0]: "Frequency" }, inplace = True)


# In[12]:


freq_plot = freq_plot.sort_values(by = ['Frequency'], ascending = False)


# In[ ]:


freq_top = freq_plot[:15]
freq_top = freq_top.reset_index()


# In[14]:


freq_index = list(freq_top['index'])

for i in range(len(freq_index)):
    freq_index[i] = freq_index[i].replace('Whatsapp_Messenger', 'W-app')
    
for i in range(len(freq_index)):
    freq_index[i] = freq_index[i].replace('Tools', 'Phone Opt')


# In[ ]:


freq_top['index'] = freq_index
names = np.unique(freq_top['index'])


# In[23]:


index = ['W-app W-app', "W-app W-app W-app", "Phone_Opt W-app", 
                 "W-app, Phone_Tools", "Phone_Tools W-app", "W-app W-app W-app W-app",
                 "Camera W-app", "W-app Camera", "W-app Dialer", "Dialer W-app", "Phone_Opt Phone_Tools",
                 "Phone_Opt W-app W-app", "Youtube W-app", "Messaging W-app", "W-app Phone_Opt"]


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
fig, plot = plt.subplots()
plot.barh(freq_top['index'], freq_top['Frequency'])
plot.set_xlabel("Total number of occurrences")
plot.set_title('Top 15 most frequent features SPM model')
#plt.gcf().subplots_adjust(left=0.32)
plot.set_ylim(len(names)-0.5, -0.5)
plt.savefig('frequency_plot_sequences_SPM.png', bbox_inches='tight')


# In[133]:


## train test splits


# In[6]:


xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 2019)


# In[ ]:





# In[ ]:





# ##### SVM model

# In[11]:


model = svm.SVC(random_state= 2019)


# In[12]:


## tuning
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'kernel': ['rbf']},
]
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)


# In[ ]:


clf.fit(xTrain, yTrain)  
print('Best score for data1:', clf.best_score_) 
print('Best C:',clf.best_estimator_.C) 
print('Best Kernel:',clf.best_estimator_.kernel)


# In[ ]:


## Cross validation
model = svm.SVC(random_state= 2019, C = 10, kernel = 'rbf')
scores = cross_val_score(model, xTrain, yTrain, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 200))


# In[ ]:





# In[ ]:


## testing on test set
model = svm.SVC(random_state= 2019, C = 10, kernel = 'rbf')
model.fit(xTrain, yTrain)


# In[243]:


y_pred = model.predict(xTest)


# In[ ]:


accuracy = accuracy_score(yTest, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100))


# In[ ]:





# ##### XGBoost model

# In[264]:


model = XGBClassifier(random_state = 2019, max_depth = 2, n_estimators = 100)


# In[41]:


## tuning 
optimization_dict = {'max_depth': [2,4,6],
                     'n_estimators': [50,100,150]}
model = GridSearchCV(model, optimization_dict, 
                     scoring='accuracy', verbose=1)
#model.fit(xTrain,yTrain)
print(model.best_score_)
print(model.best_params_)


# In[ ]:





# In[13]:


## cross validation


# In[ ]:


model = XGBClassifier(random_state = 2019, max_depth = 2, n_estimators = 100)
scores = cross_val_score(model, xTrain, yTrain, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 200))


# In[ ]:





# In[15]:


## testing on test set


# In[ ]:


model = XGBClassifier(random_state = 2019, max_depth = 2, n_estimators = 100)
model.fit(xTrain, yTrain)


# In[ ]:


y_pred = model.predict(xTest)
accuracy = accuracy_score(yTest, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100))


# In[247]:


import numpy as np
y_pred_df = pd.DataFrame(y_pred)
y_pred_df[0].value_counts()
labels = np.unique(yTest)


# In[248]:


import matplotlib.pyplot as mplt
import scikitplot as skp
from sklearn.metrics import confusion_matrix 


# In[ ]:


BoA_cm = confusion_matrix(yTest, y_pred)

BoA_cm = skp.metrics.plot_confusion_matrix(yTest, y_pred, normalize=True, title = 'Normalised Confusion Matrix SPM model with binarised stress levels')
BoA_cm.set_ylim(len(labels)-0.5, -0.5)
mplt.savefig('CM for SPM model BINARISED.png', bbox_inches='tight')


# In[ ]:





# ## Recall

# In[39]:


from sklearn.metrics import recall_score


# In[ ]:





# In[ ]:


recall = recall_score(yTest, y_pred, average = 'macro')
recall


# ## Feature importance

# In[283]:


from xgboost import plot_importance


# In[ ]:


import matplotlib.pyplot as mplt
plot_importance(model, max_num_features = 15, title = 'Top 15 feature importance SPM model')
mplt.savefig('Feature Importance SPM model .png', bbox_inches='tight')
mplt.show()


# In[ ]:





# ## Feature importance binarised SPM model

# In[3]:


spm_daily_final = pd.read_csv("spm_daily_final08_6.csv")


# In[4]:


spm_daily_final.loc[spm_daily_final.stressed < 2, 'stressed'] = 0
spm_daily_final.loc[spm_daily_final.stressed > 1, 'stressed'] = 1


# In[5]:


spm_daily_final = spm_daily_final.iloc[:,5:]


# In[6]:


x = spm_daily_final.iloc[:, :-1]
y = spm_daily_final[spm_daily_final.columns[-1]]


# In[7]:


xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 2019)


# In[ ]:


model = XGBClassifier(random_state = 2019, max_depth = 2, n_estimators = 100)
model.fit(xTrain, yTrain)


# In[ ]:


from xgboost import plot_importance
import matplotlib.pyplot as mplt
ax = plot_importance(model, max_num_features = 15, title = 'Top 15 feature importance SPM model with binarised stress levels')
#fig.set_size_inches(8,8)
mplt.savefig('Feature Importance SPM model BINARISED.png', bbox_inches='tight')
mplt.show()


# In[310]:





# In[303]:





# In[ ]:





# In[ ]:




