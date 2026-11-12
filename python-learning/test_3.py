import scipy.sparse as sp

N = 2
H = sp.csr_matrix((1 << N,1 << N))
print(H)
