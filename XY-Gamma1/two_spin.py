import my_package.spin_multibody as spin
import scipy.sparse as sp

# 参数设置
J = 1
gamma = 0.6
Gamma = 0.6
alpha = -0.5
h = 1

# 格点数设置
N = 8

# 求解哈密顿量的本征值和本征态
def Hamiltonian(J,gamma,Gamma,alpha,h):
    H = sp.csr_matrix((1 << N,1 << N),dtype=complex)
    for j in range(1,N):
        H += (
                J * (1 + gamma) / 2 * spin.two_site_kron(spin.sigma_x, j, spin.sigma_x, j+1, N)
                + J * (1 - gamma) / 2 * spin.two_site_kron(spin.sigma_y, j, spin.sigma_y, j+1, N)
                + Gamma * spin.two_site_kron(spin.sigma_x, j, spin.sigma_y, j+1, N)
                + Gamma * alpha * spin.two_site_kron(spin.sigma_y, j, spin.sigma_x, j+1, N)
                + h * spin.one_site_kron(spin.sigma_z, j, N)
             )

    H += (
                J * (1 + gamma) / 2 * spin.two_site_kron(spin.sigma_x, N, spin.sigma_x, 1, N)
                + J * (1 - gamma) / 2 * spin.two_site_kron(spin.sigma_y, N, spin.sigma_y, 1, N)
                + Gamma * spin.two_site_kron(spin.sigma_x, N, spin.sigma_y, 1, N)
                + Gamma * alpha * spin.two_site_kron(spin.sigma_y, N, spin.sigma_x, 1, N)
                + h * spin.one_site_kron(spin.sigma_z, N, N)
             )

    energies, states = sp.linalg.eigsh(H, k=1, which="SA")

    E0 = energies[0]        # 基态能量
    psi0 = states[:, 0]     # 对应的基态向量
    return E0, psi0

E0, psi0 = Hamiltonian(J,gamma,gamma,alpha,h)
print(E0)