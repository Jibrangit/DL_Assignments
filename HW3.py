import numpy as np

X_tre = np.load("fashion_mnist_train_images.npy")/255
X_tre_portion = X_tre[0:48000, :]
X_tr = X_tre_portion
X_te = np.load("fashion_mnist_test_images.npy")

labels_te = np.load("fashion_mnist_test_labels.npy")
labels_tre = np.load("fashion_mnist_train_labels.npy")
labels_tre_portion = labels_tre[0:48000]
y_tr = labels_tre_portion

X_tr = X_tre_portion

def z_k(x, w, b):
    z = np.dot(x, w) + b*np.ones(n)
    return z

def e_z_k(z):
    e_z_k = np.exp(z)
    return e_z_k

def grad_w(x, y, y_hat):
    grad_w = (-1/len(y))*(x*(y-y_hat))
    return grad_w


if __name__ == '__main__':
    m = 784
    c =10
    n = 48000
    W = np.random.randn(m, c)
    b = np.random.randn(c, 1)

    np.random.seed(42)
    np.random.shuffle(X_tr) #Not sure if shuffled indices match
    np.random.shuffle(y_tr)

    z = np.zeros([n, c])
    e_z = np.zeros([n, c])
    e_z_sum = np.zeros([n, 1])
    Y_hat = np.zeros([n, 10])
    grad_W = np.zeros([n, 10])

    print("Shape of X_tr, z, W and b: ",
          np.shape(X_tr),
          np.shape(z),
          np.shape(W),
          np.shape(b))
        
    for l in [0, c-1]:
        z[:, l] = z_k(X_tr, W[:, l], b[l])                         #Computation of all zk, where k= l = no. of classes
        e_z[:, l] = e_z_k(z[:, l])                                 #Computation of e^(z_k) for all zk
        e_z_sum[:, 0] += e_z[:, l]                                 #Summing over all e^zk
        Y_hat[:, l] = np.dot(e_z[:, l], 1/e_z_sum)                 #Computation of y_hat for all k
        grad_W = grad_w(np.transpose(X_tr), y_tr, Y_hat[:, 0])     #Calculating first gradient for all k classes
