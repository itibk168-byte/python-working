import scipy.sparse as sp

I = sp.eye(2, format="csr", dtype=complex)
sigma_x = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
sigma_y = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)


def kron_all(operators):
    result = sp.csr_matrix(operators[0])
    for operator in operators[1:]:
        result = sp.kron(result, operator, format="csr")
    return result


def one_site_kron(A, m, N):
    if not 1 <= m <= N:
        raise ValueError("m 必须在 1 到 N 之间")

    operators = [I] * N
    operators[m - 1] = sp.csr_matrix(A)
    return kron_all(operators)


def two_site_kron(A, m, B, n, N):
    if not (1 <= m <= N and 1 <= n <= N):
        raise ValueError("m 和 n 必须在 1 到 N 之间")


    if m == n:
        raise ValueError("m 和 n 必须是两个不同的格点")

    operators = [I] * N
    operators[m - 1] = sp.csr_matrix(A)
    operators[n - 1] = sp.csr_matrix(B)
    return kron_all(operators)

if __name__ == "__main__":
    print(one_site_kron(I, 2, 3))
    print(two_site_kron(I, 1, sigma_y, 2, 3))