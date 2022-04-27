#!/usr/bin/env python
# coding: utf-8

# In[1]:


import price_optim
import numpy as np
import pandas as pd
from pathlib import Path


# ## Instantiate main object

# In[2]:


price_model = price_optim.PriceOptim()


# ## Load data

# In[3]:


prod_data = pd.read_excel(Path(price_model.path_data / 'raw/Product_data.xlsx'), sheet_name="Data")
trans_data = pd.read_excel(Path(price_model.path_data / 'raw/Transaction_data.xlsx'), sheet_name="Data")


# ## Preparing and cleaning data

# In[4]:


merged_data = price_model.data_process_cww(prod_data, trans_data)


# ## Resampling original data

# In[5]:


syn_data = price_model.data_synthesis(merged_data)


# ## Additional data preparation, including aggregating data by week, generating engineered columns, appending synthetic star rating and sentiment columns, and adding google trend data for items in the subdepartment of interest and the colour of items considered.

# In[6]:


processed_data = price_model.feature_add(syn_data)


# ## Demand prediction

# In[7]:


demand_matrix, prices, sum_prices = price_model.predict_demand(processed_data)


# ## Price optimization

# In[ ]:


optimal_prices, revenue_prediction = price_model.lp_mip_solver(demand_matrix, np.array(prices), sum_prices)
print('Real Prices:' + str(price_model.actual_prices) + '\nReal Revenue: ' + str(price_model.total_revenue) + '\nPredicted Prices :' + str(optimal_prices) + '\nPredicted Revenue: ' + str(revenue_prediction))
