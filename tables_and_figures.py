#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import latex
import seaborn as sns


# In[42]:


## xgb
support4 = [0.9,0.8,0.7,0.6]
support5 = [0.9,0.8,0.7]
support6 = [0.9,0.8]
SVMmaxlength_4 = [28.07, 29.37, 28.86, 30.02]
SVMmaxlength_5 = [28.07, 29.15, 30.10]
SVMmaxlength_6 = [28.07, 30.86]

## svm
XGBmaxlength_4 = [29.68, 30.89, 30.33, 30.51]
XGBmaxlength_5 = [29.68, 31.54, 31.32]
XGBmaxlength_6 = [29.68, 32.70]


# In[43]:


font = {'size' : 17}


# In[46]:


#plt.hold(True)
fig, spag = plt.subplots()
spag.plot(support4, SVMmaxlength_4, 'b', label = "Maxlength 4")
spag.plot(support5, SVMmaxlength_5, 'g', label = "Maxlength 5")
spag.plot(support6, SVMmaxlength_6, 'r', label = "Maxlength 6")
spag.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
#spag.set_xlim(0.91, 0.59) 
spag.set_ylim(28, 33)
spag.grid(False)
spag.set_facecolor('white')
ttl = spag.title
ttl.set_position([.5, 1.05])

plt.title("SVM results different cSPADE parameters")
plt.legend(frameon=False)
plt.ylabel('Accuracy')
plt.xlabel('Support')
#plt.rc('font', **font)
plt.savefig('SVM_Accuracy_results_cSPADE.png', bbox_inches='tight')
plt.show()


# In[47]:


#plt.hold(True)
fig, spag = plt.subplots()
spag.plot(support4, XGBmaxlength_4, 'b', label = "Maxlength 4")
spag.plot(support5, XGBmaxlength_5, 'g', label = "Maxlength 5")
spag.plot(support6, XGBmaxlength_6, 'r', label = "Maxlength 6")
spag.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
#spag.set_xlim(0.91, 0.59) 
spag.set_ylim(28, 33)
spag.grid(False)
spag.set_facecolor('white')
ttl = spag.title
ttl.set_position([.5, 1.05])
plt.title("XGB results different cSPADE parameters")
plt.legend(frameon=False)
plt.rc('font', **font)
plt.ylabel('Accuracy')
plt.xlabel('Support')
plt.savefig('XGB_Accuracy_results_cSPADE.png', bbox_inches='tight')
plt.show()


# In[ ]:





# ## Apps without category plot

# In[6]:


top_10 = pd.read_csv("top_10_without_category.csv")
top_10.at[9, 'apps_without_category'] = 'IKeyboard'
top_10


# In[7]:


labels = top_10["Freq"]
names = top_10['apps_without_category']


# In[8]:


fig, plot = plt.subplots()
plot.barh(top_10['apps_without_category'], top_10['Freq'])
plot.set_xlabel("Total number of occurrences")
plot.set_title('Top 10 most frequent apps without category')
plot.grid(False)
plot.set_facecolor('white')
plot.set_ylim(len(names)-0.5, -0.5)
#plot.text(x = names, y = labels, s = labels, size = 6)
plt.savefig('Apps_without_category.png', bbox_inches='tight')


# In[ ]:





# ## Zipfian distribution for apps

# In[6]:


zipfian_for_apps = pd.read_csv("zipfian_for_apps.csv")
apps_range = range(len(zipfian_for_apps))


# In[7]:


zipfian_for_apps["range"] = apps_range


# In[21]:


fig, plot = plt.subplots()
plot.bar(zipfian_for_apps['range'], zipfian_for_apps["Freq"])
plot.set_xlim(-.5, 50) 
plot.set_ylim(0, 130000) 
plot.set_facecolor('white')
plot.grid(False)
plt.title("Frequency distribution 50 most frequent apps")
plt.gcf().subplots_adjust(left=0.15)
plt.ylabel('Total frequency')
plt.xlabel('App count')
plt.savefig('Zipf_distribution_Apps.png', bbox_inches='tight')


# In[ ]:





# ## Zipfian for categories

# In[9]:


zipfian_for_cat = pd.read_csv("zipfian_for_categories.csv")
cat_range = range(len(zipfian_for_cat))


# In[10]:


zipfian_for_cat["range"] = cat_range


# In[22]:


fig, plot = plt.subplots()
plot.bar(zipfian_for_cat['range'], zipfian_for_cat["Freq"])
plot.set_xlim(-0.5, 50) 
plot.set_ylim(0,130000) 
plot.set_facecolor('white')
plot.grid(False)
plt.title("Frequency distribution categories")
plt.gcf().subplots_adjust(left=0.15)
plt.ylabel('Total frequency')
plt.xlabel('Category count')
plt.savefig('Zipf_distribution_Categories.png', bbox_inches='tight')


# In[ ]:





# ## Zipfian for hybrid categories

# In[3]:


zipfian_for_hybrid = pd.read_csv("zipfian_after_rearranging.csv")
zipfian_for_hybrid = zipfian_for_hybrid.replace(102063, 105868)
hybrid_range = range(len(zipfian_for_hybrid))


# In[4]:


zipfian_for_hybrid["range"] = hybrid_range


# In[5]:


fig, plot = plt.subplots()
plot.bar(zipfian_for_hybrid['range'], zipfian_for_hybrid["Freq"])
plot.set_xlim(-0.5, 50) 
plot.set_ylim(0, 130000)
plot.set_facecolor('white')
plot.grid(False)
plt.title("Frequency distribution hybrid categories")
plt.gcf().subplots_adjust(left=0.15)
plt.ylabel('Total frequency')
plt.xlabel('Apps count')
plt.savefig('Zipf distribution hybrid.png')


# In[ ]:





# ## Frequency table ADASYN training

# In[6]:


data = { 'Stress level' : [0,1,2,3,4,5], 
        'Original training set' : [348, 310, 334, 125, 73, 9], 
        'ADASYN training set SPM' : [355, 313, 335, 116, 71, 352], 
        'ADASYN training set BoA' : [348, 315, 334, 121, 76, 351]}


# In[30]:


ada_freq = pd.DataFrame(data)
width = 0.22       


# In[48]:


ada_freq


# In[39]:


fig, ax = plt.subplots(figsize = (6,4))
ax.bar(ada_freq['Stress level'], ada_freq['Original training set'], width, label='Original training set')
ax.bar(ada_freq['Stress level'] + width, ada_freq['ADASYN training set SPM'], width,
    label='ADASYN training set SPM')
ax.bar(ada_freq['Stress level'] + (width*2), ada_freq['ADASYN training set BoA'], width,
    label='ADASYN training set BoA')
plt.ylabel('Amount of records in the training set')
plt.xlabel('Stress level')
plt.title('Distribution of original and ADASYN training sets')
plt.xticks(ada_freq['Stress level'] + (width*2) / 2, (0,1,2,3,4,5))
#chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
ax.legend(loc='lower left', fancybox=True, bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

plt.savefig('ADASYN_training_set_disribution.png')


# In[17]:


lineplt = plt.subplot()
lineplt.plot(ada_freq['Stress level'], ada_freq['Original training set'], '.-', label='Original training set')
lineplt.plot(ada_freq['Stress level'], ada_freq['ADASYN training set SPM'], '.-',
    label='ADASYN training set SPM')
lineplt.plot(ada_freq['Stress level'], ada_freq['ADASYN training set BoA'],'.-' , 
    label='ADASYN training set BoA')
plt.ylabel('Amount of records in the training set')
plt.xlabel('Stress level')
plt.title('Distribution of original and ADASYN training sets')
plt.xticks(ada_freq['Stress level'], (0,1,2,3,4,5))
#chartBox = ax.get_position()
#ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
ax.legend(loc='lower left')#, bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
plt.savefig('ADASYN_training_set_disribution.png')


# In[ ]:




