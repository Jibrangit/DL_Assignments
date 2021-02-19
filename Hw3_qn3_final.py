#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Load the dataset
X_tr = np.load("fashion_mnist_train_images.npy")
#print(X_tr.shape)
ytr = np.load("fashion_mnist_train_labels.npy")
print(ytr)
X_te = np.load("fashion_mnist_test_images.npy")
#print(X_te.shape)
yte = np.load("fashion_mnist_test_labels.npy")
#print(yte.shape)


# In[3]:


#Splitting the training dataset into mini-batch as training set  and validation set
split = int(0.80*X_tr.shape[0])
X_tr_mini_batch = X_tr[:split,:]
X_tr_mini_batch /=255    # normalizing the input 
#print(X_tr_mini_batch.shape)
split_y = int(0.80*ytr.shape[0])
Y_tr_mini_batch = ytr[:split_y].reshape(split_y,1)
#print(Y_tr_mini_batch.shape)

X_val = X_tr[split:,:]
X_val /= 255     #normalizing the input 
#print(X_val.shape)

Y_val = ytr[split_y:].reshape(12000,1)
#print(Y_val.shape)


# In[4]:


#converting the class labels into 1 
y_tr = np.zeros([48000, 10])
#print("initi", Y_tr_mini_batch[10:20])
for i in range(0, np.size(Y_tr_mini_batch) - 1):
    y_tr[i, Y_tr_mini_batch[i]] = 1
    
#print("fin",y_tr[10:20, :])

y_val = np.zeros([12000, 10])
#print("initi", Y_tr_mini_batch[10:20])
for i in range(0, np.size(Y_val) - 1):
    y_val[i, Y_val[i]] = 1
    
#print(y_val.shape)

y_te = np.zeros([10000, 10])
#print("initi", Y_tr_mini_batch[10:20])
for i in range(0, np.size(yte) - 1):
    y_te[i, yte[i]] = 1
    
#print("fin",y_te[10:20, :])


# In[5]:


#intialize w and b as random
examp = X_tr_mini_batch.shape[0]
pixel = X_tr_mini_batch.shape[1]
from numpy import random
w = np.random.rand(pixel,10)
b = np.random.rand(1)
#print(w.shape,b.shape)


# In[17]:


def stochastic_gradient_descent(X_tr_mini_batch, y_tr,w, b, l_reg = 0.7,batch_size = 50, learning_rate = 0.8, epoch = 100):
    n = X_tr_mini_batch.shape[0]
    rounds = int (n/batch_size)
    e = 1
    r = 1

# random shuffling of both arrays
    randomize = np.arange(len(X_tr_mini_batch))
    np.random.shuffle(randomize)
    Xtr = X_tr_mini_batch[randomize]
    Ytr = y_tr[randomize]
    
    for e in range(epoch): 
        for r in range(rounds):
    
            z = np.dot(Xtr, w) + b
            c = z - np.max(z)              # to avoid z overflow problem.
            y_hat = np.exp(c)/(np.sum(np.exp(c)))  
            print(y_hat[1,:])
            dw = (1/ n) *(np.dot(Xtr.T, Ytr - y_hat)) + ((l_reg*(w))/2) # calculating the SGD
            db = (1/ n) * (np.sum(Ytr - y_hat))

            w = w - learning_rate*dw       #Updating the values of w with l2 regularization and b 
            b = b - learning_rate*db

            param = {"w": w,
                     "b": b}

            
    
            return param


# In[18]:


def Cross_entropy_loss (param,X_val, y_val, alpha = 0.7):
    w = param["w"]
    b = param["b"]
    
    z = np.array(np.dot(X_val, w) + b)
    c = z - np.max(z)
    y_hat = np.exp(c)/(np.sum(np.exp(c)))
    print(y_hat)
    n = X_val.shape[0]
    
    import math
    fce = (- 1 / 120000) *(np.sum(np.sum(y_val*np.log(y_hat)))) + (alpha*(np.sum(np.dot(w.T,w))))/2 # Calculating the cost
    acc = format(100 - np.mean(np.abs(y_hat - y_val)) * 100)
    
    val = {"fce": fce,
           "acc": acc}
    return val


# In[ ]:


m1 = stochastic_gradient_descent(X_tr_mini_batch, y_tr,w, b, l_reg = 0.7,batch_size = 50, learning_rate = 0.8, epoch = 100)
case1 = Cross_entropy_loss(m1,X_val, y_val, alpha = 0.7)
print("case1: ",case1)


# In[14]:


m2 = stochastic_gradient_descent(X_tr_mini_batch, y_tr,w, b, l_reg = 0.02,batch_size = 100, learning_rate = 0.7, epoch = 200)
case2 = Cross_entropy_loss(m2,X_val, y_val, alpha = 0.5)
print(case2)


# In[15]:


m3 = stochastic_gradient_descent(X_tr_mini_batch, y_tr,w, b, l_reg = 0.05,batch_size = 200, learning_rate = 0.9, epoch = 300)
case3 = Cross_entropy_loss(m3,X_val, y_val, alpha = 0.7)
print(case3)


# In[16]:


m4 = stochastic_gradient_descent(X_tr_mini_batch, y_tr,w, b, l_reg = 0.08,batch_size = 150, learning_rate = 0.08, epoch = 250)
case4 = Cross_entropy_loss(m4,X_val, y_val, alpha = 0.7)
print(case4)


# In[ ]:




