import numpy as np

def one_hot_encoding(Y, NO_OF_CLASSES):
    Y_ENCODED = np.zeros([NO_OF_CLASSES, np.size(Y)])
    print(np.shape(Y_ENCODED))
    for i in range(0, np.size(Y)):
        Y_ENCODED[Y[i], i] = 1
    return Y_ENCODED


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
        np.random.seed(3)
        WT_VARS_LIST['W'+str(i+1)] = np.random.randn(NODES_PER_LAYER[i+1],NODES_PER_LAYER[i])  #Drawing from gaussian distribution within [0,1]
        BIAS_VARS_LIST['b'+str(i+1)] = np.random.randn(NODES_PER_LAYER[i+1],1)

    return WT_VARS_LIST, BIAS_VARS_LIST

def relu_function(z):
    h = np.maximum(z, 0)
    return h

def exponent_and_exponent_sum(z):
    z = np.max(z) - z
    exponent = np.exp(z)
    c = np.shape(z)
    REPEATS = c[0]
    exponent_sum = np.sum(np.exp(z), axis=0)
    exponent_sum_repeat = np.reshape(np.tile(exponent_sum, REPEATS), c)
    print('Exponent and its sum is: ', exponent[:, 1:4], exponent_sum_repeat[:, 1:4])

    return exponent_sum, exponent_sum_repeat

def softmax_function(z):
    ez, ez_sum = exponent_and_exponent_sum(z)
    exponent_sum_inverse = np.power(ez_sum, -1)
    Softmax = np.multiply(ez, exponent_sum_inverse)
    return Softmax

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

    print(np.shape(Z['z' + str(NO_OF_HIDDEN_LAYERS + 1)]))
    Y_HAT = softmax_function(Z['z' + str(NO_OF_HIDDEN_LAYERS + 1)])

    Normal_loss = (np.sum(np.multiply(Y_TRAINING, np.log(Y_HAT))))  #*(-1/NO_OF_EXAMPLES)
    # print(np.shape(Normal_loss))
    Loss = Normal_loss + regularization_loss(LAMBDA, WEIGHTS, NO_OF_EXAMPLES)

    return Z, H, Y_HAT, Loss

# X_tr = np.transpose(np.load('Training_data_randomized.npy'))
# Y_tr = np.load('Training_labels_randomized.npy')
NO_OF_EXAMPLES = 5
NUM_CLASSES = 10
X_training_validation = np.transpose(np.load('fashion_mnist_train_images.npy'))/255
Y_training_validation = np.load('fashion_mnist_train_labels.npy')
Y_training_validation_ENCODED = one_hot_encoding(Y_training_validation, NUM_CLASSES)

X_tr = X_training_validation[:, 0:NO_OF_EXAMPLES]
Y_tr = Y_training_validation_ENCODED[:, 0:NO_OF_EXAMPLES]
# X_tr = X_tr[:, 1:10]
# Y_tr = Y_tr[1:10, :]
x_shape = np.shape(X_tr)
y_shape = np.shape(Y_tr)
print('Dimension of training set: ', x_shape)
print('Dimension of training labels set: ', y_shape)
print('Last element of Y', Y_tr[NO_OF_EXAMPLES-1])

m = x_shape[0]
c = y_shape[0]
print('Weight vector size and number of classes:', m, c, '\n')
# print('Sample of training labels: ', Y_tr[30:35, :])

if __name__ == '__main__':

    NUM_HIDDEN_LAYERS = 2
    NUM_LAYERS = NUM_HIDDEN_LAYERS + 2
    NODES_PER_LAYER = list_of_nodes_per_layer(NUM_LAYERS, m, c)
    INITIAL_WTS, INITIAL_BIASES = init_wts_biases(NODES_PER_LAYER)

    print('DEBUGGING INFO....')
    # print('Nodes per layer in different layers and its shape: ', list_of_nodes_per_layer(NUM_LAYERS, m, c), np.size(NODES_PER_LAYER))

    # SAMPLE_X = X_tr[:, 1]
    # print('Sample X is: ', SAMPLE_X)

    # SAMPLE_W = INITIAL_WTS['W1']  # Just checking if seed is working
    # print('Sample W and its shape: ', SAMPLE_W[4:9, 3:6], np.shape(SAMPLE_W), '\n')
    #
    # SAMPLE_B = INITIAL_BIASES['b3']  # Just checking if seed is working
    # print('Sample b and its shape: ', SAMPLE_B[1:5], np.shape(SAMPLE_B), '\n')

    Z, H, Y_HAT, LOSS = forward_propagation(INITIAL_WTS, INITIAL_BIASES, X_tr, Y_tr, NUM_HIDDEN_LAYERS, 0.1)

    # print('Y_hat and its shape: ', Y_HAT, np.shape(Y_HAT), '\n')

    # SAMPLE_Z = Z['z3']
    # print('Sample Z and its shape: ', SAMPLE_Z, np.shape(SAMPLE_Z), '\n')

    # SAMPLE_H = H['h2']
    # print('Relu and its shape: ', SAMPLE_H, np.shape(SAMPLE_H), '\n')

    # print(len(Z), len(H), np.shape(Y_HAT), LOSS)
    # print(np.shape(INITIAL_WTS['W3']))
    print('Sample exponent value: ', np.exp(-1114.14))

    print('Loss: ', LOSS)

    # W2 = INITIAL_WTS['W2']  # Just checking if seed is working
    # print(W2[4:9, 3:6])

    Z, H, Y_HAT, LOSS = forward_propagation(INITIAL_WTS, INITIAL_BIASES, X_tr, Y_tr, NUM_HIDDEN_LAYERS, 0.6)
    print(len(Z), len(H), np.shape(Y_HAT), LOSS)
    # print(np.shape(INITIAL_WTS['W3']))



