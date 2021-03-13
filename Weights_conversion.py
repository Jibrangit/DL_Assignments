import numpy as np
np.random.seed(42)
W = np.random.randn(3, 3, 1, 64)
X = np.random.randn(784, 1)

#Generates indices where kernel has to be applied to get feature maps
Nos = np.arange(784)
Nos = np.reshape(Nos, [28, 28])
Nos = Nos[:, :-2]
Nos = np.ndarray.flatten(Nos)
print(Nos)

Kernels = {}
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

WEIGHTS = {}
WTS = []


