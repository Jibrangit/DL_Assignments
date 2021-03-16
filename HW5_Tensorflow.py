import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2
                        "Dress",        # index 3
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6
                        "Sneaker",      # index 7
                        "Bag",          # index 8
                        "Ankle boot"]   # index 9

# Image index, you can pick any number between 0 and 59,999
img_index = 56
# y_train contains the lables, ranging from 0 to 9
label_index = y_train[img_index]
# Print the label, for example 2 Pullover
print ("y = " + str(label_index) + " " +(fashion_mnist_labels[label_index]))

#Data conversion for usage
x_train = x_train.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

def visualize(X, title):
    print('Img before reshaping: ',X)
    size = np.size(X)
    X = np.reshape(X, [size, 1])
    length = int(np.sqrt(np.size(X)))
    New_img = np.reshape(X, [length, length])
    print('Img after reshaping: ',New_img)
    plt.imshow(New_img)
    plt.title(title)
    plt.show()

x_training_reshaped = np.reshape(x_train[img_index], [784, 1])
print('Shape of x_training sent to numpy is: ', np.shape(x_training_reshaped))
# # Show one of the images from the training dataset
visualize(x_train[img_index], 'Image going into model')
print('=================Tensorflow ends, Numpy begins==========================')

W = np.load('W1.npy')
X_tr = x_training_reshaped
b1 = np.load('B1.npy')
Weight2 = np.load('W2.npy')
bias2 = np.load('B2.npy')
Weight3 = np.load('W3.npy')
bias3 = np.load('B3.npy')

def conv2D(W, X, b):
    z = np.dot(W, X) +b
    return z

def maxpool(Z, width, stride):
    img_length = int(np.sqrt(len(Z)))                   #=26 in our case
    img = np.reshape(Z, [img_length, img_length])     #=13 in our case
    pooled_img_length = int(img_length/stride)

    Pooled_img = np.empty([pooled_img_length, pooled_img_length])
    for i in range(0, pooled_img_length):
        for j in range(0, pooled_img_length):
            pixels = np.array([img[i*width, j*width], img[i*width, (j*width)+1], img[(i*width)+1, j*width], img[(i*width)+1, (j*width)+1]])
            Pooled_img[i, j] = np.max(pixels)

    return Pooled_img

def relu_function(z):
    h = np.maximum(z, 0)
    return h

def dense(H, W, b):
    z = np.dot(np.transpose(W), H) + b
    return z

def softmax(z):
  return np.exp(z)/np.sum(np.exp(z), axis = 0, keepdims = True)

Kernels = {}
Wts_row = {}
WEIGHTS_CONV2D = {}
Z_CONV2D = {}
Z_MAXPOOL = {}
Z_RELU1={}

# print(W)
W1 = W[0, 0, 0, :]
W2 = W[0, 1, 0, :]
W3 = W[0, 2, 0, :]
W4 = W[1, 0, 0, :]
W5 = W[1, 1, 0, :]
W6 = W[1, 2, 0, :]
W7 = W[2, 0, 0, :]
W8 = W[2, 1, 0, :]
W9 = W[2, 2, 0, :]

for i in range(0, 64):
    Kernels['K'+str(i)] = [[W1[i], W2[i], W3[i]],
                           [W4[i], W5[i], W6[i]],
                           [W7[i], W8[i], W9[i]]]

# print(Kernels['K'+str(42)])

for i in range(0, 64):
    V = Kernels['K' + str(i)]
    Zeros = np.zeros(25)
    Wts_row['Wts' + str(i)] = np.concatenate([V[0], Zeros, V[1], Zeros, V[2]])

Wts_row_length = np.size(Wts_row['Wts' + str(56)])      #All 'weights' rows are of same length
# print('Length of weights row: ', Wts_row_length)

#Generates indices where kernel has to be applied to get feature maps
Nos = np.arange(784)                      #Slicing of last 2 columns of indices of the image as the kernel doesnt reach there.
Nos = np.reshape(Nos, [28, 28])
Nos = Nos[:-2, :-2]
Nos = np.ndarray.flatten(Nos)
print(np.shape(Nos))

for j in range(0, 64):
    WEIGHTS_CONV2D['w_conv' + str(j)] = np.concatenate([Wts_row['Wts' + str(j)], np.zeros(784 - Wts_row_length)])
    for i in range(1, 676):
        INDEX = Nos[i]
        ROW = np.concatenate([np.zeros(INDEX), Wts_row['Wts' + str(j)], np.zeros(784 - Wts_row_length - INDEX)])
        WEIGHTS_CONV2D['w_conv' + str(j)] = np.vstack([WEIGHTS_CONV2D['w_conv' + str(j)], ROW])

# print(np.shape(WEIGHTS))
# plt.matshow(WEIGHTS/np.max(WEIGHTS))
# plt.show()

for i in range(0, 64):
    Wi = WEIGHTS_CONV2D['w_conv' + str(i)]
    Z_CONV2D['z_conv2d'+str(i)] = conv2D(Wi, X_tr, b1[i])
visualize(Z_CONV2D['z_conv2d5'], 'Convolution')

for i in range(0, 64):
    Z_MAXPOOL['z_maxpool'+str(i)] = maxpool(Z_CONV2D['z_conv2d'+str(i)], 2, 2)
visualize(Z_MAXPOOL['z_maxpool5'], 'Maxpool')

for i in range(0, 64):
    Z_RELU1['z_relu_one'+str(i)] = relu_function(Z_MAXPOOL['z_maxpool'+str(i)])
visualize(Z_RELU1['z_relu_one5'], 'First_relu')

Z_FLATTENED = np.ndarray.flatten(Z_RELU1['z_relu_one'+str(0)])
for i in range(1, 64):
    Z_FLATTENED = np.concatenate([Z_FLATTENED, np.ndarray.flatten( Z_RELU1['z_relu_one'+str(i)])])
visualize(Z_FLATTENED, 'Fully flattened')

# print(np.shape(Z_FLATTENED))
Z_DENSE1 = dense(Z_FLATTENED, Weight2, bias2)
visualize(Z_DENSE1,'First fully connected')

Z_RELU2 = relu_function(Z_DENSE1)
visualize(Z_RELU2, 'Relu of first fully connected')

Z_DENSE2 = dense(Z_RELU2, Weight3, bias3)

SOFTMAX = softmax(Z_DENSE2)
print('Y_hat of Example#',img_index,' is:',SOFTMAX)
print('Y_hat label =', np.argmax(SOFTMAX), '\nItem = ',fashion_mnist_labels[np.argmax(SOFTMAX)])