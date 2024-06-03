from spin_ham import *
from qiskit.quantum_info import SparsePauliOp
import scipy, numpy
from trotter import *

for n in [7, 8, 9, 10]:
    print('n = ', n)
    # r = 1
    t = 0.5
    J, h = 1, 0.2
    hnn = Nearest_Neighbour_1d(n, Jx=0, Jy=0, Jz=J, hx=h, hy=0, hz=0, pbc=False, verbose=False)
    ob = SparsePauliOp.from_sparse_list([('Z', [0], 1)], n).to_matrix()
    exact_U = scipy.linalg.expm(-1j * t * sum([h.to_matrix() for h in hnn.ham_par]))
    h_list = [h.to_matrix(True) for h in hnn.ham_par]
    for r in [1, 2]:
        appro_U = pf_r(h_list, t, r, order=2)
        exact_ob_s = exact_U.conj().T @ ob @ exact_U 
        appro_ob_s = appro_U.conj().T @ ob @ appro_U
        # scipy.linalg.norm(exact_ob_s - appro_ob_s, ord=2)  
        # print(np.linalg.norm(exact_ob_s))
        # print(np.linalg.norm(appro_ob_s, ord=2))
        # print(np.linalg.norm(exact_ob_s - appro_ob_s, ord='nuc'))
        # print(np.trace(exact_ob_s - appro_ob_s))
        print(np.linalg.norm(exact_ob_s - appro_ob_s, ord=2))
        # np.linalg.eigvals(exact_ob_s - appro_ob_s)
        print(np.sort(abs(np.linalg.eigvalsh(exact_ob_s - appro_ob_s)))[-1])