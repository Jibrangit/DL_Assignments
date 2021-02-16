import numpy as np

X_tre = np.load("fashion_mnist_train_images.npy")/255
X_tr = np.random.shuffle(X_tre[0:48000, :])
X_te = np.load("fashion_mnist_test_images.npy")

labels_te = np.load("fashion_mnist_test_labels.npy")
labels_tr = np.load("fashion_mnist_train_labels.npy")

def z_k(x, w, b):
    z = np.dot(x, w) + b*np.eye(n, 1)
    return z

if __name__ == '__main__':
    m = 784
    c =10
    n = 48000
    W = np.random.randn(m, c)
    b = np.random.randn(c, 1)
    z = np.empty([n, 1])

    for l in range(1, c):
        z[l] = z_k(X_tr, W[0:m, l], b[l])




