#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log
import matplotlib.pyplot as plt
df = pd.read_csv('tab.csv')
plt.plot(df['d'],df['ve'],color="red",label='Validation error')
plt.plot(df['d'],df['te'],color="blue",label='Training error')
plt.xlabel ('Number of Nodes')
plt.ylabel ('Error')
plt.legend()
plt.show()


# In[ ]:




