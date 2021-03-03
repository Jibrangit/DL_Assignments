import numpy as np

X_tr = np.transpose(np.load('Training_data_randomized.npy'))
Y_tr = np.load('Training_labels_randomized.npy')

X_tr = X_tr[:, 1:100]
Y_tr = Y_tr[1:100, :]

x_shape = np.shape(X_tr)
y_shape = np.shape(Y_tr)
m = x_shape[0]
c = y_shape[1]
print('Sample of training labels: ', Y_tr[30:35, :])

def list_of_nodes_per_layer(NUM_LAYERS, NO_OF_WEIGHTS, NO_OF_CLASSES):
    l=[]
    seed_layer = (NUM_LAYERS-2) * 20
    l.append(NO_OF_WEIGHTS)
    l.append(seed_layer)
    for i in range(2, NUM_LAYERS-1):
        l.append(l[i-1] - 20)

    l.append(NO_OF_CLASSES)
    return l

def init_wts_biases(NODES_PER_LAYER):
    WT_VARS_LIST = {}
    BIAS_VARS_LIST = {}

    for i in range(0, len(NODES_PER_LAYER)-1):
        np.random.seed(42)
        WT_VARS_LIST['W'+str(i+1)] = np.random.normal(0.5, 0.5, [NODES_PER_LAYER[i+1],NODES_PER_LAYER[i]])  #Drawing from gaussian distribution within [0,1]
        BIAS_VARS_LIST['b'+str(i+1)] = np.random.randn(NODES_PER_LAYER[i+1],1)

    return WT_VARS_LIST, BIAS_VARS_LIST

def relu_function(z):
    z_shape = np.shape(z)
    h = np.empty(z_shape)
    for i in range(0, z_shape[0]-1):
        for j in range(0, z_shape[1]-1):
            if z[i, j] <=0:
                h[i, j] = 0
            else:
                h[i, j] = z[i, j]
    return h

def softmax_function(z):
    z = z - np.max(z)     #For overflow error
    z = np.array(z, dtype= np.float128)
    exponent = np.exp(z)
    exponent_sum = np.sum(np.exp(z), axis=0)
    exponent_sum_inverse = np.power(exponent_sum, -1)
    # print(exponent)
    # print(exponent_sum)
    # print(exponent_sum_inverse)
    print('Softmax: ', np.multiply(exponent, exponent_sum_inverse))
    return exponent

def regularization_loss(LAMBDA, W, NO_OF_EXAMPLES):
    W_sum = 0
    for i in range(1, len(W)+1):
       W_sum = W_sum + np.sum(np.square(W['W'+ str(i)]))

    reg_loss = (LAMBDA * 0.5 * (W_sum))/NO_OF_EXAMPLES
    return reg_loss


def forward_propagation(WEIGHTS, BIASES, X_TRAINING, Y_TRAINING, NO_OF_HIDDEN_LAYERS, LAMBDA):
    Z = {}
    H = {}
    x_shape = np.shape(X_TRAINING)
    NO_OF_EXAMPLES = x_shape[1]
    # Z['z'+str(1)] = np.dot(WEIGHTS['W1'] , X_TRAINING) + BIASES['b'+str(1)]
    # z1 = Z['z1']
    # H['h' + str(1)] = relu_function(z1)
    H['h' + str(0)] = X_TRAINING
    for i in range(0, (NO_OF_HIDDEN_LAYERS + 1)):
        Z['z' + str(i+1)] = np.dot(WEIGHTS['W'+str(i+1)], H['h'+str(i)]) + BIASES['b' + str(i+1)]
        H['h' + str(i + 1)] = relu_function(Z['z' + str(i + 1)])

    print(np.shape(Z['z' + str(7)]))
    Y_HAT = softmax_function(Z['z' + str(NO_OF_HIDDEN_LAYERS + 1)])

    Normal_loss = (np.sum(np.multiply(np.transpose(Y_TRAINING), np.log(Y_HAT))))   #*(-1/NO_OF_EXAMPLES)
    # print(np.shape(Normal_loss))
    Loss = Normal_loss + regularization_loss(LAMBDA, WEIGHTS, NO_OF_EXAMPLES)

    return Z, H, Y_HAT, Loss

if __name__ == '__main__':

    NUM_HIDDEN_LAYERS = 6
    NUM_LAYERS = NUM_HIDDEN_LAYERS + 2
    NODES_PER_LAYER = list_of_nodes_per_layer(NUM_LAYERS, m, c)
    INITIAL_WTS, INITIAL_BIASES = init_wts_biases(NODES_PER_LAYER)

    # W2 = INITIAL_WTS['W2']  # Just checking if seed is working
    # print(W2[4:9, 3:6])

    Z, H, Y_HAT, LOSS = forward_propagation(INITIAL_WTS, INITIAL_BIASES, X_tr, Y_tr, NUM_HIDDEN_LAYERS, 0.6)
    print(len(Z), len(H), np.shape(Y_HAT), LOSS)
    # print(np.shape(INITIAL_WTS['W3']))



