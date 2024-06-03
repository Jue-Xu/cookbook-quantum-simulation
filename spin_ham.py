############################
# define spin Hamiltonians
############################

import numpy as np
from qiskit.quantum_info import SparsePauliOp


class Nearest_Neighbour_1d:
    def __init__(self, n: int, Jx=1, Jy=1, Jz=1, hx=0.2, hy=0, hz=0, pbc=False, verbose=False):
        self.n = n
        self.xx_tuples = [('XX', [i, i + 1], Jx) for i in range(0, n-1)]
        self.yy_tuples = [('YY', [i, i + 1], Jy) for i in range(0, n-1)]
        self.zz_tuples = [('ZZ', [i, i + 1], Jz) for i in range(0, n-1)]
        self.x_tuples = [('X', [i], hx) for i in range(0, n)] 
        self.y_tuples = [('Y', [i], hy) for i in range(0, n)] 
        self.z_tuples = [('Z', [i], hz) for i in range(0, n)] 
        if pbc: 
            self.xx_tuples.append(('XX', [n-1, 0], Jx))
            self.yy_tuples.append(('YY', [n-1, 0], Jy))
            self.zz_tuples.append(('ZZ', [n-1, 0], Jz))

        self.ham = SparsePauliOp.from_sparse_list([*self.xx_tuples, *self.yy_tuples, *self.zz_tuples, *self.x_tuples, *self.y_tuples, *self.z_tuples], num_qubits=n).simplify() 
        self.xyz_group()
        self.par_group()
        if verbose: 
            print('The Hamiltonian: \n', self.ham)
            print('The xyz grouping: \n', self.ham_xyz)
            print('The parity grouping: \n', self.ham_par)

    def xyz_group(self):
        self.x_terms = SparsePauliOp.from_sparse_list([*self.xx_tuples, *self.x_tuples], num_qubits=self.n).simplify()
        self.y_terms = SparsePauliOp.from_sparse_list([*self.yy_tuples, *self.y_tuples], num_qubits=self.n).simplify()
        self.z_terms = SparsePauliOp.from_sparse_list([*self.zz_tuples, *self.z_tuples], num_qubits=self.n).simplify()
        self.ham_xyz = [self.x_terms, self.y_terms, self.z_terms]

    def par_group(self):
        self.even_terms = SparsePauliOp.from_sparse_list([*self.xx_tuples[::2], *self.yy_tuples[::2], *self.zz_tuples[::2], *self.x_tuples[::2], *self.y_tuples[::2], *self.z_tuples[::2]], num_qubits=self.n).simplify()
        self.odd_terms = SparsePauliOp.from_sparse_list([*self.xx_tuples[1::2], *self.yy_tuples[1::2], *self.zz_tuples[1::2], *self.x_tuples[1::2], *self.y_tuples[1::2], *self.z_tuples[1::2]], num_qubits=self.n).simplify()
        self.ham_par = [self.even_terms, self.odd_terms]


class Power_law:
    def __init__(self, n: int, alpha=4, Jx=1, Jy=1, Jz=1, hx=0.0, hy=0.0, hz=0.2, pbc=False, verbose=False):
        self.n, self.alpha = n, alpha
        self.xx_tuples = [('XX', [i, j], Jx*abs(i-j)**(-alpha)) for i in range(0, n-1) for j in range(i+1, n)]
        self.yy_tuples = [('YY', [i, j], Jy*abs(i-j)**(-alpha)) for i in range(0, n-1) for j in range(i+1, n)]
        self.zz_tuples = [('ZZ', [i, j], Jz*abs(i-j)**(-alpha)) for i in range(0, n-1) for j in range(i+1, n)]
        self.x_tuples = [('X', [i], hx) for i in range(0, n)] 
        self.y_tuples = [('Y', [i], hy) for i in range(0, n)] 
        self.z_tuples = [('Z', [i], hz) for i in range(0, n)] 
        if pbc: 
            self.xx_tuples.append(('XX', [n-1, 0], -Jx))
            self.yy_tuples.append(('YY', [n-1, 0], -Jy))
            self.zz_tuples.append(('ZZ', [n-1, 0], -Jz))

        self.ham = SparsePauliOp.from_sparse_list([*self.xx_tuples, *self.yy_tuples, *self.zz_tuples, *self.x_tuples, *self.y_tuples, *self.z_tuples], num_qubits=n).simplify()
        if verbose: print('The Hamiltonian: \n', self.ham)
        self.xyz_group()

    def xyz_group(self):
        self.ham_xyz = [SparsePauliOp.from_sparse_list([*self.xx_tuples, *self.x_tuples], num_qubits=self.n).simplify(), SparsePauliOp.from_sparse_list([*self.yy_tuples, *self.y_tuples], num_qubits=self.n).simplify(), SparsePauliOp.from_sparse_list([*self.zz_tuples, *self.z_tuples], num_qubits=self.n).simplify()]

    def parity_group(self):
        print('todo')
        return self.ham.to_matrix().todense()

class TF_Ising_1d:
    def __init__(self, n: int, J=1, h=0.2, g=0.0, pbc=False, verbose=False):
        self.n = n
        self.zz_tuples = [('ZZ', [i, i + 1], -J) for i in range(0, n-1)]
        self.x_tuples = [('X', [i], -h) for i in range(0, n)] 
        self.z_tuples = [('Z', [i], -g) for i in range(0, n)] 
        if pbc: self.zz_tuples.append(('ZZ', [n-1, 0], -J))

        self.ham = SparsePauliOp.from_sparse_list([*self.zz_tuples, *self.x_tuples, *self.z_tuples], num_qubits=n).simplify()
        if verbose: print('The Hamiltonian: \n', self.ham)
        self.parity_group()
        self.xyz_group()

    def parity_group(self):
        # return self.ham.to_matrix().todense()
        self.ham_parity = [SparsePauliOp.from_sparse_list([*self.zz_tuples[::2], *self.x_tuples[::2], *self.z_tuples[::2]], num_qubits=self.n).simplify(), SparsePauliOp.from_sparse_list([*self.zz_tuples[1::2], *self.x_tuples[1::2], *self.z_tuples[1::2]], num_qubits=self.n).simplify()]

    def xyz_group(self):
        self.ham_xyz = [SparsePauliOp.from_sparse_list([*self.zz_tuples, *self.z_tuples], num_qubits=self.n).simplify(), SparsePauliOp.from_sparse_list([*self.x_tuples], num_qubits=self.n)]

class Heisenberg_1d:
    def __init__(self, n: int, Jx=1, Jy=1, Jz=1, h=0.2, pbc=False, verbose=False):
        self.xx_tuples = [('XX', [i, i + 1], -Jx) for i in range(0, n-1)]
        self.yy_tuples = [('YY', [i, i + 1], -Jy) for i in range(0, n-1)]
        self.zz_tuples = [('ZZ', [i, i + 1], -Jz) for i in range(0, n-1)]
        self.x_tuples = [('X', [i], -h) for i in range(0, n)] 
        if pbc: 
            self.xx_tuples.append(('XX', [n-1, 0], -Jx))
            self.yy_tuples.append(('YY', [n-1, 0], -Jy))
            self.zz_tuples.append(('ZZ', [n-1, 0], -Jz))

        self.ham = SparsePauliOp.from_sparse_list([*self.xx_tuples, *self.yy_tuples, *self.zz_tuples, *self.x_tuples], num_qubits=n)
        if verbose: print('The Hamiltonian: \n', self.ham)

    def parity_group(self):
        print('todo')
        return self.ham.to_matrix().todense()
