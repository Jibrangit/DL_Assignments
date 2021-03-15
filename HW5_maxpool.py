import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
W = np.random.randn(3, 3, 1, 64)
X_tr = np.random.randn(784, 1)
b1 = np.random.randn(64, 1)
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



Kernels = {}
Wts_row = {}
WEIGHTS_CONV2D = {}
Z_CONV2D = {}
Z_MAXPOOL = {}

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
Nos = np.arange(784)
Nos = np.reshape(Nos, [28, 28])
Nos = Nos[:, :-2]
Nos = np.ndarray.flatten(Nos)
# print(Nos)

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

for i in range(0, 64):
    Z_MAXPOOL['z_maxpool'+str(i)] = maxpool(Z_CONV2D['z_conv2d'+str(i)], 2, 2)

