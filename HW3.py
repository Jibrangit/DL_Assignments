import numpy as np

X_tre = np.load("fashion_mnist_train_images.npy")/255
X_tre_portion = X_tre[0:48000, :]
X_tr = X_tre_portion
X_te = np.load("fashion_mnist_test_images.npy")

labels_te = np.load("fashion_mnist_test_labels.npy")
labels_tr = np.load("fashion_mnist_train_labels.npy")

def z_k(x, w, b):
    z = np.dot(x, w) + b*np.ones(n)
    return z

if __name__ == '__main__':
    m = 784
    c =10
    n = 48000
    W = np.random.randn(m, c)
    b = np.random.randn(c, 1)
    np.random.seed(42)
    np.random.shuffle(X_tr)
    z = np.empty([n, c])

    print(np.shape(X_tr),
          np.shape(z),
          np.shape(W),
          np.shape(b))

    for l in range(0, c-1):
        z[:, l] = z_k(X_tr, W[:, l], b[l])


