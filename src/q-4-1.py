#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log
import matplotlib.pyplot as plt
df = pd.read_csv('decision_Tree/train.csv')
df = df.sample(n=80)
f1 = df[df['left']==1]['satisfaction_level']
f2 = df[df['left']==1]['last_evaluation']
plt.scatter(f1,f2,color="red",label="Yes") 

g1 = df[df['left']==0]['satisfaction_level']
g2 = df[df['left']==0]['last_evaluation']
plt.scatter(g1,g2,color="blue",label="No")
plt.xlabel ('satisfaction_level')
plt.ylabel ('last_evaluation')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




