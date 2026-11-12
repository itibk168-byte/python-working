import scipy.sparse as sp

I = sp.eye(2,format='csr',dtype='complex')
sigma_x = sp.csr_matrix([[0, 1], [1, 0]],dtype='complex')
print(I)