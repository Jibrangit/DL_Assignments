import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.special import softmax
from sklearn.decomposition import PCA

def one_hot_encoding(Y, NO_OF_CLASSES):
    Y_ENCODED = np.zeros([NO_OF_CLASSES, np.size(Y)])
    for i in range(0, np.size(Y)):
        Y_ENCODED[Y[i], i] = 1
    return Y_ENCODED

def shuffle(X, Y):
    Xt = np.transpose(X)
    Yt = np.transpose(Y)

    np.random.seed(42)
    randomize = np.arange(len(Xt))
    np.random.shuffle(randomize)
    Xtr_transpose = Xt[randomize]
    Ytr_transpose = Yt[randomize]
    X_tr = np.transpose(Xtr_transpose)
    Y_tr = np.transpose(Ytr_transpose)
    return X_tr, Y_tr

def visualize(X_tr, Y_tr):
    X1 = X_tr[:,1000]
    Y1 = Y_tr[:,1000]
    print('Label of example1: ', Y1)
    X1 = np.reshape(X1, [28, 28])
    plt.matshow(X1)
    plt.show()

def init_wts_biases(NODES_PER_LAYER):
    WT_VARS_LIST = {}
    BIAS_VARS_LIST = {}
    WT_plot = {}
    BIASES_plot = {}

    for i in range(0, len(NODES_PER_LAYER)-1):
        np.random.seed(0)
        WT_VARS_LIST['W'+str(i+1)] =  2*(np.random.random(size=(NODES_PER_LAYER[i+1], NODES_PER_LAYER[i]))/NODES_PER_LAYER[i]**0.5) - 1./NODES_PER_LAYER[i]**0.5
        # WT_VARS_LIST['W'+str(i+1)] = np.random.normal(0, 1, [NODES_PER_LAYER[i+1],NODES_PER_LAYER[i]])#Drawing from gaussian distribution within [0,1]
        # BIAS_VARS_LIST['b'+str(i+1)] = np.ones([NODES_PER_LAYER[i+1],1])
        BIAS_VARS_LIST['b' + str(i + 1)] = 0.01 * np.ones([NODES_PER_LAYER[i+1], 1])
        flat_wts = np.reshape(WT_VARS_LIST['W'+str(i+1)], [1, np.size(WT_VARS_LIST['W'+str(i+1)])])
        repeat_biases = np.repeat(BIAS_VARS_LIST['b' + str(i + 1)], axis=0)
        flat_biases = np.reshape(repeat_biases, [1, np.size(repeat_biases)])
        WT_plot['W'+str(i+1)] =  flat_wts
        BIASES_plot['b' + str(i + 1)]  = flat_biases

    return WT_VARS_LIST, BIAS_VARS_LIST, WT_plot, BIASES_plot

def list_of_nodes_per_layer(NUM_LAYERS, NO_OF_WEIGHTS, NO_OF_CLASSES):
    l=[]
    seed_layer = (NUM_LAYERS-2) * 20
    l.append(NO_OF_WEIGHTS)
    l.append(seed_layer)
    for i in range(2, NUM_LAYERS-1):
        l.append(l[i-1] - 20)

    l.append(NO_OF_CLASSES)
    return l

def relu_function(z):
    h = np.maximum(z, 0)
    return h

def softmax_function(z):
    # z = z / np.max(np.abs(z))
    # Softmax = np.exp(z)/np.sum(np.exp(z), axis=0)                            #Works for 1 example, but will have to check use for n examples
    Softmax = softmax(z, axis=0)
    return Softmax

def regularization_loss(LAMBDA, W, NO_OF_EXAMPLES):
    W_sum = 0
    for i in range(1, len(W) + 1):
       W_sum = W_sum + np.sum(np.square(W['W'+ str(i)]))

    reg_loss = (LAMBDA * 0.5 * (W_sum))/NO_OF_EXAMPLES
    return reg_loss

def forward_propagation(WEIGHTS, BIASES, X_TRAINING, Y_TRAINING, NO_OF_HIDDEN_LAYERS, LAMBDA):
    Z = {}
    H = {}
    x_shape = np.shape(X_TRAINING)
    NO_OF_EXAMPLES = x_shape[1]
    PROBABLY_ILLEGAL = 0.0000000000001
    H['h' + str(0)] = X_TRAINING
    for i in range(0, (NO_OF_HIDDEN_LAYERS + 1)):
        Z['z' + str(i+1)] = np.dot(WEIGHTS['W'+str(i+1)], H['h'+str(i)]) + BIASES['b' + str(i+1)]
        H['h' + str(i + 1)] = relu_function(Z['z' + str(i + 1)])

    # print(np.shape(Z['z' + str(NO_OF_HIDDEN_LAYERS + 1)]))
    Y_HAT = softmax_function(Z['z' + str(NO_OF_HIDDEN_LAYERS + 1)])
    # print('Z before preprocessing it for softmax(original z): ', Z['z' + str(NO_OF_HIDDEN_LAYERS + 1)])

    Normal_loss = (-1/NO_OF_EXAMPLES) * (np.sum(np.multiply(Y_TRAINING, np.log(Y_HAT+PROBABLY_ILLEGAL))))  #*(-1/NO_OF_EXAMPLES)
    # print(np.shape(Normal_loss))
    Loss = Normal_loss + regularization_loss(LAMBDA, WEIGHTS, NO_OF_EXAMPLES)

    return Z, H, Y_HAT, Loss

def backward_propagation(Y_HAT, Y_TRAINING, WEIGHTS, BIASES, H, NUM_LAYERS, LAMBDA):
    #As a general rule, Last index of H    = NUM_LAYERS - 2
    #                   Last index of Z    = NUM_LAYERS - 1
    #                   Last index of W, b = NUM_LAYERS - 1
    dJ_dYhat = np.true_divide(Y_TRAINING, (Y_HAT))
    diagonal_elements = np.multiply(Y_HAT, 1 - Y_HAT)


    dYhat_dzL = diagonal_elements
    # print(dYhat_dzL)

    g_initial = dJ_dYhat                      #As g dimensions are cx1 initially. [Gradient is transpose of jacobian]
    g_l = np.multiply(g_initial, dYhat_dzL)   #Multiplication takes place as follows: every element of the g vector multiplies(broadcasting over all colums) with element of every 'vector' of the gradient
    # print(g_initial, g_l)

    dJ_db_l = g_l
    dJ_dw_l = np.dot(g_l, np.transpose(H['h' + str(len(H)-2)]))        #(A) Divide by m
    # print('h' + str(len(H)))
    # print('Last layer W and b gradients and their respective shapes are: ', dJ_dw_l, dJ_db_l, np.shape(dJ_dw_l), np.shape(dJ_db_l))

    # print('Size of w used to compute g before it enters for', np.shape(WEIGHTS['W' + str(NUM_LAYERS - 1)]))
    g  = np.dot(np.transpose(WEIGHTS['W' + str(NUM_LAYERS - 1)]), g_l)
    # print('Shape of g before it enters for loop', np.shape(g))

    dJ_dW = {}
    dJ_dB = {}

    dJ_dW['dJ_dw' + str(NUM_LAYERS - 1)] = dJ_dw_l
    dJ_dB['dJ_db' + str(NUM_LAYERS - 1)] = dJ_db_l

    for r in range((NUM_LAYERS-2), 0, -1):
        h = H['h' + str(r)]                                         #Eg: h6
        h_previous = H['h' + str(r-1)]                              #Eg: h5
        w = WEIGHTS['W' + str(r)]                                   #Eg: w6
        dH_dZ = np.where(h>0, 1, 0)
        g = np.multiply(g, dH_dZ)

        dJ_dB['dJ_db' + str(r)] = g
        dJ_dW['dJ_dw' + str(r)] = np.dot(g, np.transpose(h_previous)) + (LAMBDA * w)
        # print('Dimension of w and b gradients for loop#:', r, np.shape(dJ_dW['dJ_dw' + str(r)]), np.shape(dJ_dB['dJ_db' + str(r)]))

        g = np.dot(np.transpose(w), g)
        # print(np.shape(g))

    return dJ_dW, dJ_dB

def Stochastic_Gradient_Descent(X_tr, Y_tr, Learning_rate, DECAY_LEARNING_RATE, Regularization_constant, NUM_EPOCHS, BATCH_SIZE, NO_OF_LAYERS):
    NO_OF_HIDDEN_LAYERS = NO_OF_LAYERS - 2
    DATASET_SIZE = len(np.transpose(Y_tr))
    print('Size of dataset: ', DATASET_SIZE)
    WEIGHT_VEC_SIZE = len(X_tr)
    CLASS_VEC_SIZE = len(Y_tr)
    NODES_PER_LAYER = list_of_nodes_per_layer(NO_OF_LAYERS, WEIGHT_VEC_SIZE, CLASS_VEC_SIZE)
    WEIGHTS, BIASES, W_plot, B_plot = init_wts_biases(NODES_PER_LAYER)
    W, B = WEIGHTS, BIASES

    for epochs in range(0, NUM_EPOCHS):
        for i in range(0, DATASET_SIZE, BATCH_SIZE):
            X = X_tr[:, (i*BATCH_SIZE):((i+1)*BATCH_SIZE)]
            Y = Y_tr[:, (i*BATCH_SIZE):((i+1)*BATCH_SIZE)]

            Z, H, Y_HAT, LOSS = forward_propagation(WEIGHTS, BIASES, X, Y, NO_OF_HIDDEN_LAYERS, Regularization_constant)
            dJdW, dJdB = backward_propagation(Y_HAT, Y, WEIGHTS, BIASES, H, NO_OF_LAYERS, Regularization_constant)

            if (i%50 == 0):
                print('On this example right now: ', i * BATCH_SIZE)
                Learning_rate = Learning_rate * DECAY_LEARNING_RATE
                print('Loss right now: ', LOSS)


            for i in range(1, NO_OF_LAYERS):
                W['W' + str(i)] = W['W' + str(i)] - Learning_rate * dJdW['dJ_dw' + str(i)]
                W_plot['W' + str(i)] = np.concatenate((W_plot['W' + str(i)], np.reshape(W['W' + str(i)], [1, np.size(W['W' + str(i)])])))
                B['b' + str(i)] = B['b'+str(i)] - Learning_rate * dJdB['dJ_db' + str(i)]
                B_plot['b'+str(i)] = np.concatenate((B_plot['b'+str(i)],np.reshape(B['b'+str(i)], [1, np.size(B['b' + str(i)])])))


    UPDATED_W, UPDATED_B = W, B
    return UPDATED_W, UPDATED_B, W_plot, B_plot

if __name__ == '__main__':

    TRAINING_SIZE = 400
    BATCH_SIZE = 1
    NUM_CLASSES = 10
    NUM_HIDDEN_LAYERS = 6

    X_training_validation = np.transpose(np.load('fashion_mnist_train_images.npy')) / 255
    Y_training_validation = np.load('fashion_mnist_train_labels.npy')
    LENGTH_OF_ENTIRE_SET = len(np.transpose(X_training_validation))
    Y_training_validation_ENCODED = one_hot_encoding(Y_training_validation, NUM_CLASSES)

    X_tr_unrandomized = X_training_validation[:, 0:TRAINING_SIZE]
    Y_tr_unrandomized = Y_training_validation_ENCODED[:, 0:TRAINING_SIZE]

    VALIDATION_INDICES =[TRAINING_SIZE, LENGTH_OF_ENTIRE_SET]
    print('Length of entire set is: ', LENGTH_OF_ENTIRE_SET)
    X_validation = X_training_validation[:, VALIDATION_INDICES[0]: VALIDATION_INDICES[1]]
    Y_validation = Y_training_validation_ENCODED[:, VALIDATION_INDICES[0]: VALIDATION_INDICES[1]]

    X_testing = np.transpose(np.load('fashion_mnist_test_images.npy')) / 255
    Y_testing = np.load('fashion_mnist_test_labels.npy')
    Y_testing_ENCODED = one_hot_encoding(Y_testing, NUM_CLASSES)

    # X_tr = X_tr[:, 1:10]
    # Y_tr = Y_tr[1:10, :]
    x_shape = np.shape(X_tr_unrandomized)
    y_shape = np.shape(Y_tr_unrandomized)
    xv_shape = np.shape(X_validation)
    yv_shape = np.shape(Y_validation)
    xt_shape = np.shape(X_testing)
    yt_shape = np.shape(Y_testing_ENCODED)

    print('Shape of unrandomized training set: ', x_shape)
    print('Shape of unrandomized training labels: ', y_shape)

    print('Shape of validation set: ', xv_shape)
    print('Shape of validation labels: ', yv_shape)

    print('Shape of testing set: ', xt_shape)
    print('Shape of testing labels: ', yt_shape)

    X_tr, Y_tr = shuffle(X_tr_unrandomized, Y_tr_unrandomized)

    LEARNING_RATE1 = 0.1
    DECAY_LEARNING_RATE1 = 0.1
    REG_CONSTANT1 = 0.6                #Lambda
    NUM_EPOCHS1 = 1

    Valid_weights, Valid_biases, PLOT_W, PLOT_B = Stochastic_Gradient_Descent(X_tr, Y_tr, LEARNING_RATE1, DECAY_LEARNING_RATE1, REG_CONSTANT1, NUM_EPOCHS1, BATCH_SIZE, NUM_HIDDEN_LAYERS + 2)

    # print(Valid_weights, Valid_biases)

    Z_V, H_V, Y_HAT_V, Loss_V =  forward_propagation(Valid_weights, Valid_biases, X_validation, Y_validation, NUM_HIDDEN_LAYERS, REG_CONSTANT1)
    print('Loss on validation set: ', Loss_V)

    Z_T, H_T, Y_HAT_T, Loss_T = forward_propagation(Valid_weights, Valid_biases, X_testing, Y_testing_ENCODED,
                                                    NUM_HIDDEN_LAYERS, REG_CONSTANT1)
    print('Loss on testing set: ', Loss_T)

    y_hat_t = np.argmax(Y_HAT_T, axis=0)
    print(y_hat_t[85:90])
    y_t = np.argmax(Y_testing_ENCODED, axis=0)
    print(y_t[85:90])
    ACCURACY = np.mean(y_hat_t==y_t)
    print('Accuracy on test data = ', ACCURACY)

    print('Shapes of W1 and B1 are', np.shape(PLOT_W['W' + str(1)]), np.shape(PLOT_B['b'+str(1)]))
