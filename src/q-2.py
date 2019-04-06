#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log
dframe = pd.read_csv('decision_Tree/train.csv')
df = dframe
arr = ['Work_accident','promotion_last_5years','sales','salary','left']
Class = 'left'


# In[24]:


def get_subtable(df,node,value,flg,k):
    if(flg==0): #numeric
        if(k==0): # <=
            return  df[df[node] <= value].drop(node,axis=1)
        else:
             return df[df[node] > value].drop(node,axis=1)
    else:
        return df[df[node] == value].drop(node,axis=1)

def find_entropy(df):
    Class = 'left'   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return abs(entropy)

def find_entropy_attribute(df,attribute):
    flg = 0;
    if(attribute not in arr): 
        arr1 = list(df[attribute].values)
#         print(attribute)
#         print(len(arr1))
#         if(len(arr1)==0):
#             print(attribute)
#             print(df)
#             return 100000
        mini = min(arr1)
        maxe = max(arr1)
        mid = (mini+maxe)/2
        #less than mid one class
        flg = 1
    
    Class = 'left'   #To make the code generic, changing target variable class name
    target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'\
    
    if flg == 0:
        variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
        entropy2 = 0
        for variable in variables: #hot cold...
            entropy = 0
            for target_variable in target_variables: #yes no
                num = len(df[attribute][df[attribute]==variable][df[Class]==target_variable]) #hot and yes
                den = len(df[attribute][df[attribute]==variable]) #only hot
                fraction = num/(den+eps) 
                entropy += -fraction*log(fraction+eps)
            fraction2 = len(df[attribute][df[attribute]==variable])/len(df)
            entropy2 += -fraction2*entropy2
    else: #numeric
        entropy2 = 0
        for k in range(2):
            den = 0
            entropy = 0
            for target_variable in target_variables: #yes no
                if k == 0:
                    num = len(df[attribute][df[attribute] <= mid][df[Class] == target_variable])
                    den = len(df[attribute][df[attribute] <= mid])
                    fraction = num/(den+eps)
                    entropy += -fraction*log(fraction+eps)
                else:
                    num = len(df[attribute][df[attribute] > mid][df[Class] == target_variable])
                    den = len(df[attribute][df[attribute] > mid])
                    fraction = num/(den+eps)
                    entropy += -fraction*log(fraction+eps)
            fraction2 = den/len(df)
            entropy2 += -fraction2*entropy
        entropy2 = entropy2/2       
    return abs(entropy2)

def find_winner(df):
    Entropy_att = []
    IG = []
    ans = float('-inf')
    out = 'left'
#     print(df.keys())
    for key in df.keys():
        if key=='left':
            continue
#         Entropy_att.append(find_entropy_attribute(df,key))
        ans1 = max(find_entropy(df)-find_entropy_attribute(df,key),ans)
        if(ans1!=ans):
            ans = ans1
            out = key
    return out

def dec_tree(df,tree=None):
    Class = 'left'
    node = find_winner(df)
#     print(node)
    
    if tree is None:                    
        tree={}
        tree[node] = {}
        root = node
    if(node not in arr):
        arr1 = list(df[node].values) 
#         print(arr1)
#         if(len(arr1)==0):
#             print(node)
#             print(df)
#             tree[node]={}
#             return tree
        mini = min(arr1)
        maxe = max(arr1)
        mid = (mini+maxe)/2
        for k in range(2):
            subtable = get_subtable(df,node,mid,0,k)
            clValue = 'left'
            if(k==0):
                value = "<=" + str(mid) 
            else:
                value = ">" + str(mid)
            df11 = subtable.left.value_counts()
            if(len(df11)==1):
                tree[node][value] = df11.keys()[0]
            elif (len(subtable.columns)==1):
                df1 = subtable.left.value_counts()
#                 if(len(df1)==0):
#                     tree[node][value] = "N.A"
                if(len(df1)==1):
                    tree[node][value] = df1.keys()[0]
                elif df1[0] > df1[1]:
                    tree[node][value] = 0
                else:
                    tree[node][value] = 1
                return tree
            else:
                tree[node][value] = dec_tree(subtable)
    else:
        attValue = df[node].unique()
        for value in attValue:
            #only table with low is o/p
            subtable = get_subtable(df,node,value,1,0)
            clValue = 'left' 
            df11 = subtable.left.value_counts()
            if(len(df11)==1):
                tree[node][value] = df11.keys()[0]
            elif (len(subtable.columns)==1):#Checking purity of subset
                df1 = subtable.left.value_counts()
                if(len(df1)==1):
                    tree[node][value] = df1.keys()[0]
                elif df1[0] > df1[1]:
                    tree[node][value] = 0
                else:
                    tree[node][value] = 1
            else:
                tree[node][value] = dec_tree(subtable) #Calling the function recursively 
    return tree,node


# In[64]:

def run_on_set(input_set,tree):
    root1 = list(tree)[0]
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    results = []
    for entry in input_set.iterrows():
        tempDict = tree
        root = root1
        result = ""
        while(1):
            if(root in arr):
                value = entry[1][root]  #low
            else:
                if(len(list(tempDict[root]))==1):
                    value  = list(tempDict[root])[0]

                elif((list(tempDict[root])[0])[2:] < str(entry[1][root])):
                    value = list(tempDict[root])[1]
                else:
                    value = list(tempDict[root])[0]
#             print(value)
            if((value in tempDict[root])==False):
                result = "Null"
                break;
            elif(tempDict[root][value]==0 or tempDict[root][value]==1):
                result = tempDict[root][value]
                break;
            else:
                result11 = tempDict[root][value]
                if(isinstance(result11,tuple)):
                    tempDict = tempDict[root][value]
                    tempDict = list(tempDict)[0]
                    root = list(result11)[1]    
                else:
                    tempDict = tempDict[root][value]
                    root = list(result11)[0]
        
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


# In[65]:


def run_decision_tree(df):
    training_set = df.sample(frac=0.8)
    test_set = df.sample(frac=0.2)
    tree,root1 = dec_tree(training_set)
    
    accuracy,Precision,Recall,F1_score = run_on_set(test_set,tree)
    
    #########################################
    accuracy1,Precision1,Recall1,F1_score1 = run_on_set(training_set,tree)
    
    
    print("Accuracy: " + str(accuracy))
    print("Recall: " + str(Recall))
    print("Precision: " + str(Precision))
    print("F1_score: " + str(F1_score))
    return (1-accuracy),(1-accuracy1)
#     acc.append(accuracy)
#     avg_acc = sum(acc)/len(acc)
#     print(avg_acc)


# In[ ]:





# In[66]:


# df = dframe[['satisfaction_level','Work_accident','promotion_last_5years','sales','salary','left']]
df = dframe
run_decision_tree(df)


# In[47]:


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
            flg = 0
            if(root in arr):
                value = entry[1][root]  #low
            else:
#                 print(tempDict[root])
                if(len(list(tempDict[root]))==1):
                    value  = list(tempDict[root])[0]

                elif((list(tempDict[root])[0])[2:] < str(entry[1][root])):
                    value = list(tempDict[root])[1]
                else:
                    value = list(tempDict[root])[0]
                
#             print(tempDict[root])
            if((value in tempDict[root])==False):
                result = "Null"
                break;
            elif((tempDict[root][value]==0 or tempDict[root][value]==1)):
                result = tempDict[root][value]
                break;
            else:
                result11 = tempDict[root][value]
                if(isinstance(result11,tuple)):
                    tempDict = tempDict[root][value]
                    tempDict = list(tempDict)[0]
                    root = list(result11)[1]    
                else:
                    tempDict = tempDict[root][value]
                    root = list(result11)[0]
#             else:
#                 result = "Null"
#                 break
        if result != "Null":
            out.append(result)
        else:
            out.append("N.A")
    return out


# In[49]:


tree,root1 = dec_tree(df)
# print(tree)
test_sample = pd.read_csv('decision_Tree/sample_test.csv')
predict(tree,test_sample)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




