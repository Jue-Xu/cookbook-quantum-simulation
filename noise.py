from qiskit.quantum_info import SparsePauliOp, DensityMatrix, shannon_entropy, Statevector, partial_trace
import numpy as np
from numpy.linalg import matrix_power, norm
import itertools as it
import jax
import scipy.linalg as la

from colorspace import sequential_hcl
purp = sequential_hcl("Purples") # Purples
red = sequential_hcl("Reds")
green = sequential_hcl("Greens")
orange = sequential_hcl("Oranges")
import matplotlib as mpl

class GradColors:
    def __init__(self, rate_num):
        self.purple = mpl.colors.ListedColormap(purp(rate_num+1)[:-1][::-1], name='from_list', N=None) 
        self.red = mpl.colors.ListedColormap(red(rate_num+1)[:-1][::-1], name='from_list', N=None) 
        self.green = mpl.colors.ListedColormap(green(rate_num+1)[:-1][::-1], name='from_list', N=None) 
        self.orange = mpl.colors.ListedColormap(orange(rate_num+1)[:-1][::-1], name='from_list', N=None) 
        # self.return_colors()

    def get_colors(self):
        return self.purple, self.red, self.green, self.orange

def my_entropy(state, base: int = 2):
    evals = np.maximum(np.real(la.eigvals(state)), 0.0)
    return shannon_entropy(evals, base=base)

class NoisyPF:
    def __init__(self, H_list, init_state, p, t, r, verbose=False, depolar_type='local', eval_ob=True):
        if isinstance(init_state, str):
            self.n = len(init_state)
            self.init_rho = DensityMatrix.from_label(init_state).to_operator().to_matrix()
        else:
            self.init_rho = init_state
            self.n = int(np.log2(self.init_rho.dim[0]))

        self.dt = t / r
        self.pf_U, self.exact_U = pf_step(H_list, self.dt)
        self.magn_op = SparsePauliOp.from_sparse_list([('Z', [i], 1.) for i in range(0, self.n)], self.n).to_matrix()
        self.z_ob = SparsePauliOp.from_sparse_list([('Z', [0], 1.)], self.n).to_matrix()
        self.x_ob = SparsePauliOp.from_sparse_list([('X', [0], 1.)], self.n).to_matrix()
        self.phy_err, self.alg_err, self.tot_err, self.acc_err = [], [], [], []
        self.magn, self.echo, self.purity, self.entropy = [], [], [], []
        # purity.append(np.trace(rho @ rho))
        self.magn_noise, self.magn_trott, self.magn_exact = [], [], []
        self.z_ob_noise, self.z_ob_trott, self.z_ob_exact = [], [], []
        self.x_ob_noise, self.x_ob_trott, self.x_ob_exact = [], [], []
        self.rho = self.init_rho
        self.psi_pf, self.psi_exact = self.init_rho, self.init_rho
        for _ in np.arange(0, r) :
            self.rho_pf = self.pf_U @ self.rho @ self.pf_U.conj().T
            self.rho_exact = self.exact_U @ self.rho @ self.exact_U.conj().T
            self.psi_pf = self.pf_U @ self.psi_pf @ self.pf_U.conj().T
            self.psi_exact = self.exact_U @ self.psi_exact @ self.exact_U.conj().T
            # rho_trott_r = matrix_power(pf_U, r+1) @ self.init_rho @ matrix_power(pf_U.conj().T, r+1)

            if depolar_type == 'global':
                self.rho = depolarize(self.rho_pf, p)
            if depolar_type == 'time':
                print('todo')
                raise NotImplementedError
            else:
                self.rho = depolarize(self.rho_pf, p, n_qubits=self.n)

            self.alg_err.append(norm(self.rho_pf - self.rho_exact, ord='nuc'))
            self.phy_err.append(norm(self.rho - self.rho_pf, ord='nuc'))
            self.tot_err.append(norm(self.rho - self.rho_exact, ord='nuc'))
            self.acc_err.append(norm(self.rho - self.psi_exact, ord='nuc'))
            # spectral norm; Frobenius norm (average case bound)

            if eval_ob:
                self.purity.append(np.trace(self.rho @ self.rho))
                self.magn_noise.append(np.trace(self.rho @ self.magn_op)/self.n)
                self.magn_trott.append(np.trace(self.psi_pf @ self.magn_op)/self.n)
                self.magn_exact.append(np.trace(self.psi_exact @ self.magn_op)/self.n)
                self.z_ob_noise.append(np.trace(self.rho @ self.z_ob))
                self.z_ob_trott.append(np.trace(self.psi_pf @ self.z_ob))
                self.z_ob_exact.append(np.trace(self.psi_exact @ self.z_ob))
                self.x_ob_exact.append(np.trace(self.psi_exact @ self.x_ob))
                self.entropy.append(my_entropy(partial_trace(DensityMatrix(np.array(self.rho)), list(range(0, self.n//2)))))
                self.echo.append(Statevector.from_label(init_state).data.conj().T @ self.psi_exact @ Statevector.from_label(init_state).data)
            # self.echo.append(np.trace(rho_exact_r @ self.init_rho)) 
            if verbose: print(f'Purity: {self.purity[-1]}')

def product_state(n):
    """
    Generate a pure product state for n qubits.

    Parameters:
    - n (int): Number of qubits.

    Returns:
    - numpy array: Pure product state for n qubits.
    """
    # Define basis states
    zero = np.array([1, 0])
    one = np.array([0, 1])

    # Initialize the product state with the first qubit state
    psi = zero

    # Generate the product state for remaining qubits
    for _ in range(n - 1):
        psi = np.kron(psi, zero)

    return psi
 
def pauli_tensor(n_qubits, pauli: str, coeff=1, to_matrix=True):
    if pauli in ['X', 'Y', 'Z']:
        if to_matrix:
            return SparsePauliOp(pauli*n_qubits, coeff).to_matrix()
        else:
            return SparsePauliOp(pauli*n_qubits, coeff)
    else:
        raise ValueError('Invalid Pauli.')

def ob_evo(rho, U, pauli=False):
    if pauli:
        return U @ rho @ U
    else:
        return U @ rho @ U.conj().T

def depolarize(rho, p, n_qubits=0, verbose=False):
    dim = rho.shape[0]
    n = int(np.log2(dim))
    # print('n_qubits: ', n_qubits)
    assert n_qubits == 0 or n_qubits == n

    if n_qubits == 0: # depolarize all qubits
        return (1-p)*rho + p*np.eye(dim)/dim
    else:
        # return (1-p)*rho + p/3*(ob_evo(rho, pauli_tensor(n_qubits, 'X')) + ob_evo(rho, pauli_tensor(n_qubits, 'Y')) + ob_evo(rho, pauli_tensor(n_qubits, 'Z')))
        all_pstrs = [''.join(i) for i in list(it.product(['I', 'X', 'Y', 'Z'], repeat=n))]
        # if verbose: print(all_pstrs)
        pstr_wt = [sum(1 for char in pstr if char != 'I') for pstr in all_pstrs]
        pstr_dict = dict(zip(all_pstrs, pstr_wt))
        if verbose: print('weight[pstr]: ', pstr_dict)

        return sum([(p/3)**pstr_dict[pstr]*(1-p)**(n-pstr_dict[pstr])*ob_evo(rho, SparsePauliOp(pstr, 1).to_matrix()) for pstr in pstr_dict.keys()])

def matrices_product(list_U):
    product = (list_U[0])
    for i in range(1, len(list_U)):
        product = product @ list_U[i]

    return product

def pf_step(H_list, dt, order=2):
    """
    Compute the time evolution of a density matrix using the
    second-order product formula.

    Parameters:
    - rho (numpy array): Initial density matrix.
    - H_list (list): List of Hamiltonians.
    - dt (float): Time.

    Returns:
    - numpy array: Density matrix at time dt.
    """
    # Compute the time evolution
    # appro_U = standard_trotter(H_list, dt, 1, ord=order)    
    if order == 2:
        list_U = [jax.scipy.linalg.expm(-1j * dt / 2 * herm.toarray()) for herm in H_list]
        forward_product = matrices_product(list_U) 
        reverse_product = matrices_product(list_U[::-1])
        appro_U = forward_product @ reverse_product
    elif order == 1:
        appro_U = matrices_product([jax.scipy.linalg.expm(-1j * dt * herm.toarray()) for herm in H_list]) 
    else:
        raise ValueError('Invalid order.')

    exact_U = jax.scipy.linalg.expm(-1j * dt * sum(H_list).toarray())

    return appro_U, exact_U
    # rho_t = appro_U @ rho @ appro_U.conj().T
    # return rho_t

def init_state(n_qubits, verbose=False):
    init_psi = product_state(n_qubits)
    rho = np.outer(init_psi, init_psi.conj())
    if verbose: print('init rho: ', rho)    

    return rho