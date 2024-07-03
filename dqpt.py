from qiskit.quantum_info import Statevector, SparsePauliOp, Operator, partial_trace, entropy, shannon_entropy, DensityMatrix
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit_algorithms import TimeEvolutionProblem, TrotterQRTE
from qiskit.synthesis import ProductFormula, LieTrotter, SuzukiTrotter
from scipy.linalg import expm
import numpy as np

from utils import lighten_color
from noise import pf_step

class ObPF:
    def __init__(self, H_list, ob, init_state, t, r, pf_order=1, verbose=False) -> None:
        self.ob, self.t, self.r, self.pf_order = ob, t, r, pf_order
        self.H_list, self.init_state = H_list, init_state
        self.n = init_state.num_qubits
        self.L = len(H_list)
        if verbose: print(f'n={self.n}, L={self.L}, t={t}, r={r}')

        self.magn_op = SparsePauliOp.from_sparse_list([('Z', [i], 1.) for i in range(0, self.n)], self.n).to_matrix()/(self.n)

        self.ob_keys = ['echo', 'rate', 'magn', 'obdt', 'entr', 'otoc']
        self.data = dict([(ob_key, {key: [] for key in ['exact', 'trott']} ) for ob_key in self.ob_keys])

        self.pf()

    def pf(self):
        self.dt = self.t/(self.r)
        self.init_rho = DensityMatrix(self.init_state).to_operator().to_matrix()
        self.pf_U, self.exact_U = pf_step(self.H_list, self.dt, order=1)
        self.psi_trott, self.psi_exact = self.init_rho, self.init_rho

        for i in range(self.r+1):
            self.data['magn']['trott'].append(np.trace(self.psi_trott @ self.magn_op))
            self.data['magn']['exact'].append(np.trace(self.psi_exact @ self.magn_op))
            self.data['echo']['trott'].append(self.init_state.data.conj().T @ self.psi_trott @ self.init_state.data)
            self.data['echo']['exact'].append(self.init_state.data.conj().T @ self.psi_exact @ self.init_state.data)

            self.psi_trott = self.pf_U @ self.psi_trott @ self.pf_U.conj().T
            self.psi_exact = self.exact_U @ self.psi_exact @ self.exact_U.conj().T
            # self.psi_exact = expm(-1j * self.dt * (i+1) * sum(self.H_list).toarray()) @ self.init_rho @ expm(1j * self.dt * (i+1) * sum(self.H_list).toarray())

def normalize(data):
    s = sum(a**2 for a in data)
    return [a**2/s for a in data]

def get_hamiltonian(L, J, h, g=0, verbose=False):
    ZZ_tuples = [('ZZ', [i, i + 1], -J) for i in range(0, L-1)]
    X_tuples = [('X', [i], -h) for i in range(0, L)]
    Z_tuples = [('Z', [i], -g) for i in range(0, L)]
    # ZZ_tuples = [('XX', [i, i + 1], -J) for i in range(0, L-1)]
    # X_tuples = [('Z', [i], -h) for i in range(0, L)]

    hamiltonian = SparsePauliOp.from_sparse_list([*ZZ_tuples, *X_tuples, *Z_tuples], num_qubits=L)
    if verbose: print('The Hamiltonian: \n', hamiltonian)
    return hamiltonian.simplify()

def ob_trott_err(ob_val_dict, abs=True):
    if abs:
        return np.abs(np.subtract(ob_val_dict['exact'], ob_val_dict['trott']))
    else:   
        return np.subtract(ob_val_dict['exact'], ob_val_dict['trott'])

def binGradSeach(T, H, init_state, ob, r_pf, dt, eps, verbose=False):
    t_c, t_d = 0, T
    t_c_temp = t_c
    t_d_temp = t_d
    track_t = [[], []]
    track_ob = [[], []]
    if verbose: print(f"t_c={t_c}, t_d={t_d}, t_c_temp={t_c_temp}, t_d_temp={t_d_temp}")
    while abs(t_c - t_d) > eps:
        track_t[0].append(t_c)
        track_t[1].append(t_d)
        _, ob_c, _, _ = dynamics1shot(H, init_state, ob, t_c, r_pf)
        _, ob_cc, _, _ = dynamics1shot(H, init_state, ob, t_c+dt, r_pf)
        _, ob_d, _, _ = dynamics1shot(H, init_state, ob, t_d, r_pf)
        _, ob_dd, _, _ = dynamics1shot(H, init_state, ob, t_d-dt, r_pf)
        track_ob[0].append(ob_c)
        track_ob[1].append(ob_d)
        print(f"ob_c={ob_c:.3f}, ob_cc={ob_cc:.3f}, ob_d={ob_d:.3f}, ob_dd={ob_dd:.3f}")
        if ob_cc > ob_c:
            if ob_dd > ob_d:
                if verbose: print('1')
                t_c_temp = t_c
                t_c = (t_c + t_d)/2
            else:
                if verbose: print('2')
                t_c_temp = t_c
                t_c = t_d
                t_d = t_d_temp
                # t_c_temp = t_c
                # t_c = (t_c + t_d)/2
        elif ob_dd > ob_d and ob_cc < ob_c:
            if verbose: print('3')
            t_d_temp = t_d
            t_d = t_c
            t_c = t_c_temp
            # t_d_temp = t_d
            # t_d = (t_c + t_d)/2
        else:
            raise ValueError('Error in binGradSeach')
        if verbose: print(f"t_c={t_c:.3f}, t_d={t_d:.3f}, t_c_temp={t_c_temp:.3f}, t_d_temp={t_d_temp:.3f}")

    return (t_c + t_d)/2, track_t, track_ob 

def dynamics(H, init_state, ob, t_list, r_pf, pf_order=1, verbose=False):
    # echo, rate, magn, entr, obdt = dict({'exact':[], 'trott':[]}), dict({'exact':[], 'trott':[]}), dict({'exact':[], 'trott':[]}), dict({'exact':[], 'trott':[]}), dict({'exact':[], 'trott':[]})
    # echo, rate, magn, entr, obdt = ({key: [] for key in ['exact', 'trott']} for _ in range(5))
    ob_keys = ['echo', 'rate', 'magn', 'obdt', 'entr', 'otoc']
    data = dict([(k, {key: [] for key in ['exact', 'trott']} ) for k in ob_keys])
    n = H.num_qubits
    magn_op = SparsePauliOp.from_sparse_list([('Z', [i], 1.) for i in range(0, n)], num_qubits=n)
    for t in t_list:
        problem = TimeEvolutionProblem(H, initial_state=init_state, time=t)
        if pf_order == 1:
            trotter = TrotterQRTE(num_timesteps=r_pf)
        else:
            trotter = TrotterQRTE(product_formula=SuzukiTrotter(order=pf_order), num_timesteps=r_pf)
        result = trotter.evolve(problem)
        trott_state = Statevector(result.evolved_state)
        exact_state = init_state.evolve(expm(-1j * t * H.to_matrix()))
        if verbose: print('evolved state (Trotter): \n', trott_state)
        data['echo']['trott'].append(abs(trott_state.data.conj().T @ init_state.data)**2)  
        data['echo']['exact'].append(abs(exact_state.data.conj().T @ init_state.data)**2)
        data['rate']['trott'].append(-np.log(data['echo']['trott'][-1])/n)  
        data['rate']['exact'].append(-np.log(data['echo']['exact'][-1])/n)  
        data['magn']['trott'].append(np.real(trott_state.expectation_value(magn_op))/n)
        data['magn']['exact'].append(np.real(exact_state.expectation_value(magn_op))/n)
        data['obdt']['trott'].append(np.real(trott_state.expectation_value(ob))/n)
        data['obdt']['exact'].append(np.real(exact_state.expectation_value(ob))/n)
        data['entr']['exact'].append(entropy(partial_trace(exact_state, list(range(0, n//2)))))
        data['entr']['trott'].append(entropy(partial_trace(trott_state, list(range(0, n//2)))))
    # return echo, rate, magn, entr, obdt
    return data

def plot_evo(ax, t_list, y_list, marker, color='', title='', xlabel='', ylabel='', label='', markersize=5, markeredgewidth=1, inset=False):
    if color == '':
        ax.plot(t_list, y_list, marker, label=label, markersize=markersize, markeredgewidth=markeredgewidth)
        # ax.plot(t_list, y_list, '-', markersize=5)
        # ax.plot(t_list, y_list, 'o', label=label, markersize=5)
        # ax.plot(t_list, y_list, marker, label=label, markeredgecolor='k', markeredgewidth=0.4, markersize=5)
    else:
        ax.plot(t_list, y_list, marker, color=color, label=label, markeredgecolor=color, markeredgewidth=markeredgewidth, markersize=markersize, mfc=lighten_color(color, 0.3))
        # ax.plot(t_list, y_list, marker, color=color, label=label, markeredgecolor=color, markeredgewidth=0.4, markersize=markersize, mfc=color[:-2]+"80")
    if not inset: 
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    if title  != '': ax.set_title(title)
    if xlabel != '': 
        ax.set_xlabel(xlabel)
    # else:
    #     ax.set_xticks([]) 
    if ylabel != '': ax.set_ylabel(ylabel)
    # else:
    #     ax.set_xticks([])

def letter_annotation(axes, xoffset, yoffset, letters):
    # https://towardsdatascience.com/a-guide-to-matplotlib-subfigures-for-creating-complex-multi-panel-figures-70fa8f6c38a4
    for letter in letters:
        axes[letter].text(xoffset, yoffset, f'({letter})', transform=axes[letter].transAxes, size=14, weight='bold')

def ob_dt(ob_list, t_list, ord=1):
    """time derivative of observable expectation 

    Args:
        ob_list (_type_): _description_
        t_list (_type_): _description_

    Returns:
        ob_dt_list: _description_
    """
    if ord == 1:
        ob_dt_list = [(ob_list[i + 1] - ob_list[i]) / (t_list[-1]/len(t_list))  for i in range(len(ob_list) - 1)]
    elif ord == 2:
        ob_dt_list = [(ob_list[i + 2] - 2*ob_list[i + 1] + ob_list[i]) / (0.5*t_list[-1]/len(t_list))  for i in range(len(ob_list) - 2)]
    return ob_dt_list

def dynamics1shot(H, init_state, ob, t, r_pf, verbose=False):
    n = H.num_qubits
    problem = TimeEvolutionProblem(H, initial_state=init_state, time=t)
    trotter = TrotterQRTE(num_timesteps=r_pf)
    result = trotter.evolve(problem)
    trott_state = Statevector(result.evolved_state)
    exact_state = init_state.evolve(expm(-1j * t * H.to_matrix()))
    if verbose: print('evolved state (Trotter): \n', trott_state)
    echo = abs(trott_state.data.conj().T @ init_state.data)**2
    echo = abs(exact_state.data.conj().T @ init_state.data)**2
    rate = -np.log(echo)/n 
    rate = -np.log(echo)/n 
    magn = np.real(trott_state.expectation_value(ob))/n
    magn = np.real(exact_state.expectation_value(ob))/n
    entr = entropy(partial_trace(exact_state, list(range(0, n//2))))
    entr = entropy(partial_trace(trott_state, list(range(0, n//2))))

    return echo, rate, magn, entr

# def plot_dist(ax, dist, color, title='', xlabel='', ylabel=''):
#     ax.bar(list(range(len(dist))), dist, color=color)
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_xticks([])
#     ax.set_label(ylabel)

# def dynamic_evo(axes, n_list, t_list, trott_step, verbose=False, comment='', inset=False):
#     t_sample = len(t_list)
#     for n in n_list:    
#         echo, echo_trott, magn_exact = [], [], []
#         H = get_hamiltonian(L=n, J=.2, h=1., verbose=True)
#         init_state = Statevector.from_label('0'*n)
#         magnetization_op = SparsePauliOp.from_sparse_list([('Y', [i], 1.) for i in range(0, n)], num_qubits=n)
#         for t in t_list:
#             # print('initial state: \n', init_state)
#             problem = TimeEvolutionProblem(H, initial_state=init_state, time=t)
#             trotter = TrotterQRTE(num_timesteps=trott_step)
#             result = trotter.evolve(problem)
#             # print(result)
#             trott_evolution = Statevector(result.evolved_state)
#             # print('evolved state (Trotter): \n', statevector)

#             exact_evolution = init_state.evolve(expm(-1j * t * H.to_matrix()))
#             echo.append(-np.log(abs(exact_evolution.data.conj().T @ init_state.data)**2)/n)  
#             echo_trott.append(-np.log(abs(trott_evolution.data.conj().T @ init_state.data)**2)/n)   
#             magn_exact.append(np.real(exact_evolution.expectation_value(magnetization_op)))

#         derivative = [(echo[i + 1] - echo[i]) / (t/t_sample)  for i in range(len(echo) - 1)]
#         derivative_trott = [(echo_trott[i + 1] - echo_trott[i]) / (t/t_sample)  for i in range(len(echo_trott) - 1)]
#         # print(derivative)
#         plot_dynamics(axes[0], t_list, echo, title=comment, ylabel='Rate function', marker='.-', label=f'n={n} (Exact)', inset=inset)
#         plot_dynamics(axes[0], t_list, echo_trott, title=comment, ylabel='Rate function', marker='-.', label=f'n={n} (Trotter)', inset=inset)
#         plot_dynamics(axes[1], t_list[1:], derivative, ylabel='Time derivative', marker='.-', label=f'n={n} (Exact)', inset=inset)
#         plot_dynamics(axes[1], t_list[1:], derivative_trott, ylabel='Time derivative', marker='-.', label=f'n={n} (Trotter)', inset=inset)
#         plot_dynamics(axes[2], t_list, magn_exact, xlabel='Evolution time', ylabel='Magnetization (Y)', marker='.-', label=f'n={n} (Exact)')
#         plot_dynamics(axes[2], t_list, [0] * len(t_list), xlabel='Evolution time', ylabel='Magnetization (Y)', marker='--', label=f'n={n} (Exact)')

#     return echo, echo_trott
