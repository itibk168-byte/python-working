import numpy as np

N=2
states = np.arange(1 << N)
H = np.zeros((1 << N, 1 << N), complex)
print(H)
print(H.dtype)