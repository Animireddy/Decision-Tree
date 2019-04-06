#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log


# In[26]:


dframe = pd.read_csv('decision_Tree/train.csv')


# In[27]:


dframe.sales.unique()


# In[28]:


df = dframe[['Work_accident','promotion_last_5years','sales','salary','left']]


# In[29]:


dframe.keys()


# In[30]:


def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return abs(entropy)
def find_giniindex(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 2
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy = entropy*fraction
    return abs(entropy)
def find_miscrate(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = float('inf')
    values = df[Class].unique()
    for value in values:
        entropy = max(entropy,df[Class].value_counts()[value]/len(df[Class]))
    return abs(entropy)


# In[31]:


def find_entropy_attribute(df,attribute):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
    variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    entropy2 = 0
    for variable in variables:  #high low
        entropy = 0
        for target_variable in target_variables: #Yes No
            num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
            den = len(df[attribute][df[attribute]==variable])
            fraction = num/(den+eps)
            entropy += -fraction*log(fraction+eps)
        fraction2 = len(df[attribute][df[attribute]==variable])/len(df)
        entropy2 += -fraction2*entropy
    return abs(entropy2)

def find_giniindex_attribute(df,attribute):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
    variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    entropy2 = 0
    for variable in variables:  #high low
        entropy = 2
        for target_variable in target_variables: #Yes No
            num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
            den = len(df[attribute][df[attribute]==variable])
            fraction = num/(den+eps)
            entropy = entropy*fraction
        fraction2 = len(df[attribute][df[attribute]==variable])/len(df)
        entropy2 += fraction2*entropy
    return abs(entropy2)

def find_miscrate_attribute(df,attribute):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
    variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    entropy2 = 0
    for variable in variables:  #high low
        entropy = float('inf')
        for target_variable in target_variables: #Yes No
            num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
            den = len(df[attribute][df[attribute]==variable])
            entropy = max(entropy,df[Class].value_counts()[target_variable]/len(df[Class]))
        fraction2 = len(df[attribute][df[attribute]==variable])/len(df)
        entropy2 += fraction2*entropy
    return abs(entropy2)


# In[32]:


def find_winner(df,im):
    Entropy_att = []
    IG = []
    ans = float('-inf')
    out = df.keys()[-1]
    for key in df.keys()[:-1]:
#         Entropy_att.append(find_entropy_attribute(df,key))
        if(im==0):
            ans1 = max(find_entropy(df)-find_entropy_attribute(df,key),ans)
        elif(im==1):
            ans1 = max(find_giniindex(df)-find_giniindex_attribute(df,key),ans)
        elif(im==2):
            ans1 = max(find_miscrate(df)-find_miscrate_attribute(df,key),ans)
        if(ans1!=ans):
            ans = ans1
            out = key
    return out


# In[33]:


def get_subtable(df,node,value):
    return df[df[node] == value].drop(node,axis=1)


# In[34]:


def dec_tree(df,im,tree=None):
    Class = df.keys()[-1]
    node = find_winner(df,im)
    
    attValue = df[node].unique()
    
    if tree is None:                    
        tree={}
        tree[node] = {}
        root = node
    for value in attValue:
        #only table with low is o/p
        subtable = get_subtable(df,node,value)

        clValue = subtable.keys()[-1] 
        df11 = subtable.left.value_counts()
        if(len(df11)==1):
            tree[node][value] = df11.keys()[0]
        elif len(subtable.columns)==1:#Checking purity of subset

            df1 = subtable.left.value_counts()
            if(len(df1)==1):
                tree[node][value] = df1.keys()[0]
            elif df1[0] > df1[1]:
                tree[node][value] = 0
            else:
                tree[node][value] = 1
        elif len(subtable.columns)==1:
            tree[node][value] = subtable
        else:        
            tree[node][value] = dec_tree(subtable,im) #Calling the function recursively 
                   
    return tree,root


# In[35]:


tree = dec_tree(df,0)


# In[12]:


import pprint
pprint.pprint(tree)


# In[36]:


def run_on_set(input_set,tree):
    results = []
    root1 = list(tree)[0]
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    for entry in input_set.iterrows():
        tempDict = tree
#         print(tempDict)
        root = root1
        result = ""
        while(1):
            value = entry[1][root]  #low
            if((value in tempDict[root])==False):
                result = "Null"
                break;
            elif(tempDict[root][value]==0 or tempDict[root][value]==1):
                result = tempDict[root][value]
                break;
            elif(value in df[root].values):
                result = tempDict[root][value]
                tempDict = tempDict[root][value]
                tempDict = list(tempDict)[0]
                root = list(list(result)[0])[0]
            else:
                result = "Null"
                break
        if result != "Null":
            if(result == entry[1]["left"]):
                if(result==0):
                    TN+=1
                else:
                    TP+=1
            else:
                if(entry[1]["left"]==0):
                    FP+=1
                else:
                    FN+=1
                
            results.append(result == entry[1]["left"])

    accuracy = float(results.count(True))/float(len(results))
    Recall = TP/(TP+FN+eps)
    Precision = TP/(TP+FP+eps)
    F1_score = 2/((1/Recall)+(1/Precision)+eps)
    return accuracy,Precision,Recall,F1_score


# In[37]:


def run_decision_tree(df,im):
#     K = 10
#     attributes = list(df.keys())
#     acc = []
#     for k in range(K):
    training_set = df.sample(frac=0.8)
    test_set = df.sample(frac=0.2)
    tree,root1 = dec_tree(training_set,im)
    accuracy,Precision,Recall,F1_score = run_on_set(test_set,tree)
    
    #########################################
    
    accuracy1,Precision1,Recall1,F1_score1 = run_on_set(training_set,tree)
    print("Accuracy: " + str(accuracy))
    print("Recall: " + str(Recall))
    print("Precision: " + str(Precision))
    print("F1_score: " + str(F1_score))
    return accuracy,accuracy1


# In[38]:


run_decision_tree(df,0)


# In[19]:


# df = dframe[['salary','left']]
# run_decision_tree(df,0)


# In[20]:


#tree = decision_tree
#model_args = attributes
#sample_test = file
def predict(tree,test_sample):
#     print(tree)
    out = []
    for entry in test_sample.iterrows():
        tempDict = tree
#         print(tempDict)
        root = list(tempDict)[0] #salary
#         print(root)
        result = ""
        while(1):
            value = entry[1][root]  #low
#             print(tempDict[root])
                
            if((value in tempDict[root])==False):
                result = "Null"
                break;
            elif(tempDict[root][value]==0 or tempDict[root][value]==1):
                result = tempDict[root][value]
                break;
            elif(value in df[root].values):
                result = tempDict[root][value]
                tempDict = tempDict[root][value]
                tempDict = list(tempDict)[0]
                root = list(list(result)[0])[0]
            else:
                result = "Null"
                break
        if result != "Null":
            out.append(result)
        else:
            out.append("N.A")
    return out


# In[15]:


tree,root1 = dec_tree(df,0)
# print(tree)
test_sample = pd.read_csv('decision_Tree/sample_test.csv')


# In[413]:

predict(tree,test_sample)


# In[ ]:




