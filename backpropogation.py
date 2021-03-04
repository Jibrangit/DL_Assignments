#!/usr/bin/env python
# coding: utf-8



def backpropogation(Y_tr,Y_HAT,NUM_LAYERS,WEIGHTS,LAMBDA):
    g = np.divide(Y_tr, y_hat)
    e = np.eye(10,10)
    g = np.multiply(g, (np.multiply(Y_HAT, e)))
   
    GDb = {}
    GDw = {}
    GDb['GDb' + str(NUM_LAYERS)] = g
    GDw['GDw' + str(NUM_LAYERS)] = np.dot(g,(h['h'+ str(NUM_LAYERS)]).T) + (LAMBDA*WEIGHTS['W' + str(NUM_LAYERS)])
    
    g = np.dot(((WEIGHTS['W' + str(NUM_LAYERS)]).T),g)
    
    for k in revrse (NUM_LAYERS-1,1):
        
        relu_dash = np.mutiply(np.where((H['h' + str(k)]) = 0, 0, 1), e) 
        
        g = np.multiply(g, (relu_dash))
        GDb['GDb' + str(k)] = g
        GDw['GDw' + str(k)] = np.dot(g,(h['h'+ str(k)]).T) + (LAMBDA*WEIGHTS['W' + str(k)])
        
        g = np.dot(((WEIGHTS['W' + str(k)]).T),g)
    
    return GDw, GDb
    

def stochastic_gradient_descent(WEIGHTS, BIASES, GDb, GDw,learning_rate,NUM_LAYERS):
    
    for i in range (1, NUM_LAYERS):
        WEIGHTS['W' + str(i)] = WEIGHTS['W' + str(i)] - learning_rate*GDw['GDw' + str(i)]
        BIASES['b' + str(i)] = BIASES['b' + str(i)] - learning_rate*GDb['GDb' + str(k)]
        
    return WEIGHTS, BIASES


def ff_bp_nn(X_tr, Y_tr, layers_dims, epoch =250, batch_size = 50, W, b):
    np.random.seed(1)
    costs = []                        
    
    n = X_tr.shape[0]
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
            # Forward propagation
            Loss, Y_HAT = forward_propagation(WEIGHTS, BIASES, X_TRAINING, Y_TRAINING, NO_OF_HIDDEN_LAYERS, LAMBDA)
       
            # Backward propagation.
            GDw, GDb = backpropogation(Y_tr,Y_HAT,NUM_LAYERS,WEIGHTS,LAMBDA)
        
            # Update WTS_BIASES.
            WEIGHTS, BIASES = stochastic_gradient_descent(WEIGHTS, BIASES, GDb, GDw,learning_rate,NUM_LAYERS)
            
    # Final_Loss
    fce_Loss = forward_propagation(WEIGHTS, BIASES, X_tr, Y_tr, NUM_HIDDEN_LAYERS, 0.6)
    
    return fce_Loss
       

