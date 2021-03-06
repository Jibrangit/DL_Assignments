#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(42)


# In[2]:


def initialization(number_of_layers = 6):
    nodes = []
    nodes.append(784)
    for n in range (1, 5):
            new_nodes = int(784 - 100*n)
            nodes.append(new_nodes)
    nodes.append(10)
    print(nodes)
    W = {}
    W['W' + str(0)] = np.random.randn(nodes[0],784)
    b = {}
    b['b' + str(0)] = np.random.randn(nodes[0], 1)

    for i in range (1,number_of_layers):
            np.random.seed(42)
            W['W' + str(i)] = np.random.randn(nodes[i], nodes[i-1])
            b['b'+ str(i)] = np.random.randn(nodes[i], 1)

    print(b)
    return W, b


# In[3]:


#Load the dataset
Xtr = np.load("fashion_mnist_train_images.npy")
#print(X_tr.shape)
ytr = np.load("fashion_mnist_train_labels.npy")
#print(ytr)
X_tet = np.load("fashion_mnist_test_images.npy")
X_te = X_tet.T
print('X_te:', X_te.shape)
yte = np.load("fashion_mnist_test_labels.npy")
Y_te = yte.T.reshape(1,10000)
print('Y_te:', Y_te.shape)


#Splitting the training dataset into mini-batch as training set  and validation set
split = int(0.80*Xtr.shape[0])
X_trt = Xtr[:split,:]
X_trt /=255    # normalizing the input 

X_tr = X_trt.T
print('X_tr:', X_tr.shape)

split_y = int(0.80*ytr.shape[0])
Y_trt = ytr[:split_y].reshape(split_y,1)

Y_tr = Y_trt.T
print('Y_tr:' ,Y_tr.shape)

X_valt = Xtr[split:,:]
X_valt /= 255     #normalizing the input 

X_val = X_valt.T
print('X_val:' ,X_val.shape)

Y_valt = ytr[split_y:].reshape(12000,1)

Y_val = Y_valt.T
print('Y_val:',Y_val.shape)


# In[4]:


def relu_function(z):
    h = np.maximum(0,z)
    return h

def softmax_function(z):
    c = z - np.max(np.abs(z))  
    softmax = np.exp(c)/(np.sum(np.exp(c), axis =0))
    return softmax


# In[5]:


def regularization_loss(LAMBDA, W, n):
    W_sum = 0
    for i in range (1, len(W)+1):
        W_sum = W_sum + np.sum(np.square(W['W'+ str(i)]))
        
    reg_loss = (LAMBDA * 0.5 * (W_sum))/n
    
    return reg_loss


# In[6]:


def feedforward (X_tr, Y_tr, W, b, number_of_layers):
    Z = {}
    H = {}
    
    H['h' + str(0)] = X_tr
    for i in range(1, number_of_layers):
        Z['z' + str(i)] = np.dot((W['W' + str(i)]),(H['h'+ str(i-1)])) + (b['b'+ str(i)])
        H['h' + str(i)] = relu_function(Z['z' + str(i)])
        
    print(np.shape(Z['z' + str(NO_OF_HIDDEN_LAYERS)]))
    
    Y_hat = softmax_function(Z['z' + str(NO_OF_HIDDEN_LAYERS)])
    
    n = X_tr.shape[1]
    Loss = (-1/n)*(np.sum(np.multiply(Y_tr, (np.log(Y_hat))))) + regularization_loss(LAMBDA, W, n)
    return Y_hat, Z, H, W,  b, Loss


# In[7]:


def backward (GDc, GDH, GDw, GDb, LAMBDA):
    m = H.shape[1] 
    
    GDw = {}
    GDb = {}
    GDH = {}
    
    GDw = (np.dot(GDc,(H).T))/m + (LAMBDA*GDw)/m
    GDb = (np.sum(GDc, axis = 1))/m
    GDH = np.dot(((GDw).T),GDc)
    
    return GDH, GDw, GDb


# In[8]:


def softmax_back(GDc):
    
    
    
    return GDH

def relu_back(GDc):
    
    if (GDc > 0):
        GDH = 1
    else:
         GDH = 0
    return GDH


# In[9]:


def backward_propogation(Y_hat, Y_tr, Z, H, W, b, number_of_layers):
    
    GDc = np.divide(Y_tr, Y_hat)
    
    m = Y_hat.shape[1]
    Y_tr = Y_tr.reshape(Y_hat.shape)
    
    GDc['GDc' + str(number_of_layers)] = np.divide(Y_tr, Y_hat)
    GDw['GDw' + str(number_of_layers)] ,GDb['GDb' + str(number_of_layers)] = backward(softmax_back(GDc,H['h' + str(number_of_layers)],W['W' + str(number_of_layers)], b['b' + str(number_of_layers)]), W['W' + str(number_of_layers -1)], b['b' + str(number_of_layers -1)] ,H['h'+ str(number_of_layers -1)])
    
    
    for i in range((number_of_layers -1), 1, -1):
        
        GDc['GDc' + str(i)] = np.divide(Y_tr, Y_hat)
        GDw['GDw' + str(i)] ,GDb['GDb' + str(i)] = backward(relu_back(GDc,GDH['h' + str(i)], W['W' + str(i)], b['b'+ str(i)]),W['W' + str(i-1)], b['b' + str(i-1)] ,GDH['h'+ str(i-1)]) 
            
            
    return GDc, GDw, GDb


# In[10]:


def sgd(GDw, GDb, W, b, learning_rate, number_of_layers):
    
    for i in range (1, number_of_layers):
        W['W' + str(i)] = W['W' + str(i)] - learning_rate*GDw['GDw' + str(i)]
        B['b' + str(i)] = B['b' + str(i)] - learning_rate*GDb['GDb' + str(i)]
        
    return W, B


# In[11]:


def ff_bp_nn(X,Y, number_of_layers = 6, learning_rate = 0.004, epoch = 250 ,batch_size = 50):
    
    W, b = initialization(number_of_layers)
    np.random.seed(1)
    costs = []                        
    
    n = X_tr.shape[0]
    rounds = int (n/batch_size)
    e = 1
    r = 1

# random shuffling of both arrays
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    Xtr = X[randomize]
    Ytr = Y[randomize]
    
    
    for e in range(epoch): 
        for r in range(rounds):
            
            Y_hat, Z, H, W,  b, Loss = feedforward(Xtr, Ytr,W, b, number_of_layers)
            
            GDw, GDb = backward_propogation(Y_hat, Y_tr, Z, H, W, b, number_of_layers)
            
            W, b = sgd(GDw, GDb, W, b, learning_rate, number_of_layers)
            
    print(cost)
    
    return Loss       


# In[ ]:





# In[ ]:




