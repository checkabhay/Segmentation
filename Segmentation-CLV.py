#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import datetime as dt


# In[3]:


data = pd.read_excel('D:\Abhay-doc\learn MA\Segmentation\Online_Retail.xlsx')
#df = pd.read_csv(r'D:\Abhay-doc\learn MA\Marketing-Mix\Advertising.csv',index_col='Sl')


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data= data[pd.notnull(data['CustomerID'])]


# In[7]:


filtered_data=data[['Country','CustomerID']].drop_duplicates()


# In[8]:


#Top ten country's customer
filtered_data.Country.value_counts()[:10].plot(kind='bar')


# In[9]:


uk_data=data[data.Country=='United Kingdom']


# In[10]:


uk_data.info()


# In[11]:


uk_data.describe()


# In[12]:


uk_data = uk_data[(uk_data['Quantity']>0)]
uk_data.info()


# In[13]:


uk_data=uk_data[['CustomerID','InvoiceDate','InvoiceNo','Quantity','UnitPrice']]


# In[14]:


uk_data['TotalPrice'] = uk_data['Quantity'] * uk_data['UnitPrice']
uk_data['InvoiceDate'].min(),uk_data['InvoiceDate'].max()


# In[15]:


PRESENT = dt.datetime(2011,12,10)
uk_data['InvoiceDate'] = pd.to_datetime(uk_data['InvoiceDate'])


# In[16]:


uk_data.head()


# In[17]:


rfm= uk_data.groupby('CustomerID').agg({'InvoiceDate': lambda date: (PRESENT - date.max()).days,
                                        'InvoiceNo': lambda num: len(num),
                                        'TotalPrice': lambda price: price.sum()})


# In[18]:


rfm.columns


# In[19]:


# Change the name of columns
rfm.columns=['monetary','frequency','recency']


# In[20]:


rfm['recency'] = rfm['recency'].astype(int)


# In[21]:


rfm.head()


# In[22]:


rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, ['1','2','3','4'])
rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['4','3','2','1'])
rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1'])


# In[23]:


rfm.head()


# In[24]:


rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
rfm.head()


# In[25]:


# Filter out Top/Best cusotmers
rfm[rfm['RFM_Score']=='111'].sort_values('monetary', ascending=False).head()


# In[26]:


rfm['z_monetary'] = (rfm['monetary'] - rfm['monetary'].mean())/rfm['monetary'].std()
rfm['z_frequency'] = (rfm['frequency'] - rfm['frequency'].mean())/rfm['frequency'].std()
rfm['z_recency'] = (rfm['recency'] - rfm['recency'].mean())/rfm['recency'].std()


# In[27]:


rfm.head()


# In[28]:


from sklearn import cluster
model = cluster.KMeans(n_clusters=4, random_state=10)
model.fit(rfm[['z_monetary','z_frequency','z_recency']])
rfm['cluster'] = model.labels_
rfm.head()


# In[33]:


rfm.to_csv (r'D:\Abhay-doc\learn MA\Segmentation\Cluster.csv', index = False, header=True)


# In[34]:


#Start of CLV Calculation 
uk_data=uk_data[['CustomerID','InvoiceDate','InvoiceNo','Quantity','UnitPrice']]


# In[43]:


# Calculate RFM_Score
rfm['RFM_Score1'] = rfm[['r_quartile','f_quartile','m_quartile']].sum(axis=1)
print(rfm['RFM_Score1'].head())
#rfm['RFM_Score1'] = rfm['r_quartile'].value_counts() + rfm['f_quartile'].value_counts()+ rfm['m_quartile'].value_counts()


# In[44]:


rfm.head()


# In[45]:


rfm.info()


# In[ ]:




