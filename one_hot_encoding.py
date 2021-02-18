import numpy as np

labels_te = np.load("fashion_mnist_test_labels.npy")
labels_tre = np.load("fashion_mnist_train_labels.npy")
labels_tre_portion = labels_tre[0:48000]
y_tr = labels_tre_portion

emp = np.zeros([48000, 10])

for i in range(0, np.size(y_tr) -1):
    emp[i, y_tr[i]] = 1
print(y_tr[0:10])
print(emp[0:10, :])
