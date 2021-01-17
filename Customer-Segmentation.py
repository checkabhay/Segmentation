#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import os
ageinc_df = pd.read_csv(r'D:\Abhay-doc\learn MA\Data-Science-for-Marketing-Analytics-master\Data-Science-for-Marketing-Analytics-master\Lesson03\ageinc.csv')


# In[5]:


ageinc_df['z_income'] = (ageinc_df['income'] - ageinc_df['income'].mean())/ageinc_df['income'].std()
ageinc_df['z_age'] = (ageinc_df['age'] - ageinc_df['age'].mean())/ageinc_df['age'].std()
os.getcwd()


# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(ageinc_df['income'], ageinc_df['age'])
plt.xlabel('Income')
plt.ylabel('Age')
plt.show()


# In[ ]:


from sklearn import cluster

model = cluster.KMeans(n_clusters=4, random_state=10)
model.fit(ageinc_df[['z_income','z_age']])


# In[ ]:


ageinc_df['cluster'] = model.labels_
ageinc_df.head()


# In[6]:


colors = ['r', 'b', 'k', 'g']
markers = ['^', 'o', 'd', 's']

for c in ageinc_df['cluster'].unique():
    d = ageinc_df[ageinc_df['cluster'] == c]
    plt.scatter(d['income'], d['age'], marker=markers[c], color=colors[c])
    
plt.xlabel('Income')
plt.ylabel('Age')
plt.show()


# In[ ]:




