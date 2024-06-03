from cmath import exp
from scipy import sparse
from scipy.linalg import expm
from numpy.linalg import matrix_power
import scipy.sparse.linalg as ssla

import numpy as np

def commutator(A, B):
    return A @ B - B @ A

def norm(A, ord='spectral'):
    if ord == 'fro':
        return np.linalg.norm(A)
    elif ord == 'spectral':
        return np.linalg.norm(A, ord=2)
    else:
        # raise ValueError('norm is not defined')
        return np.linalg.norm(A, ord=ord)

def analytic_bound(H, k, t, r):
    L = len(H)
    Lambda = max([norm(h) for h in H])

    return (2 * L * 5**(k-1) * Lambda * t)**(2*k+1)/(3*r**(2*k)) * exp((2*L*5**(k-1)*Lambda*t)/r)

def interference_bound(H, t, r):
    # Layden_2022_First-Order Trotter Error from a Second-Order Perspective
    try:
        assert len(H) == 2
    except:
        raise ValueError('The Hamiltonian contains not exactly 2 terms')

    h1 = H[0]
    h2 = H[1]
    C1 = min(norm(h1), norm(h2))
    C2 = 0.5 * norm(commutator(h1, h2))
    S = [norm(commutator(h1, commutator(h1, h2))), norm(commutator(h2, commutator(h2, h1)))]
    C3 = 1 / 12 * (min(S) + 0.5 * max(S))
    e1 = C1 * t / r
    e2 = C2 * t**2 / r
    e3 = C3 * t**3 / r**2
    bound = min(e2, e1 + e3, 2)
    # bound = min(e2, e1 + e3, 2 * len(h1))

    return bound, e1, e2, e3

def triangle_bound(h, k, t, r):
    L = len(h)
    if k == 1:
        if L == 2:
            raise ValueError('k=1 is not defined for L=2')
        elif L == 3:
            c = norm(commutator(h[0], h[1])) + norm(commutator(h[1], h[2])) + norm(commutator(h[2], h[0]))
            error = c * t**2 / (2*r) 
    return error

def tight_bound(h, order: int, t: float, r: int, type='spectral', verbose=False):
    L = len(h)
    d = h[0].shape[0]
    if order == 1:
        a_comm = 0
        for i in range(0, L):
            temp = np.zeros(h[0].shape, dtype=complex)
            for j in range(i + 1, L):
                temp += commutator(h[i], h[j])
            a_comm += norm(temp, ord=type)

        if type == 'spectral':
            error = a_comm * t**2 / (2*r)
        elif type == 'fro':
            error = a_comm * t**2 / (2*r*np.sqrt(d))
        else:
            raise ValueError(f'type={type} is not defined')
    elif order == 2:
        c1 = 0
        c2 = 0
        for i in range(0, L):
            temp = np.zeros(h[0].shape, dtype=complex)
            for j in range(i + 1, L):
                temp += h[j]
            # h_sum3 = sum(h[k] for k in range(i+1, L))
            # print(h_sum3.shape)
            # h_sum2 = sum(h[k] for k in range(i+1, L))
            c1 += norm(commutator(temp, commutator(temp, h[i])), ord=type) 
            # c1 = norm(commutator(h[0]+h[1], commutator(h[1]+h[2], h[0]))) + norm(commutator(h[2], commutator(h[2], h[1])))
            # c2 = norm(commutator(h[0], commutator(h[0],h[1]+h[2]))) + norm(commutator(h[1], commutator(h[1], h[2])))
            c2 += norm(commutator(h[i], commutator(h[i], temp)), ord=type)
        if type == 'spectral':
            error = c1 * t**3 / r**2 / 12 + c2 *  t**3 / r**2 / 24 
        elif type == 'fro':
            # print(c1, c2)
            error = c1 * t**3 / r**2 / 12 / np.sqrt(d) + c2 *  t**3 / r**2 / 24 / np.sqrt(d)
            # print('random input:', error)
        else:
            raise ValueError(f'type={type} is not defined')
    else: 
        raise ValueError(f'higer order (order={order}) is not defined')

    return error

def analytic_loose_commutator_bound_parity(n, J, h, dt, pbc=False, verbose=False):
    if pbc:
        c1 = 16*J**2*h*(n) + 4*J**2*h*(n)
        c2 = 8*(n)*J**2*h
    else:
        # c1 = 16*J**2*h*(n-1) + 4*J**2*h*(n-2)
        # c2 = 8*(n-1)*J**2*h
        if n % 2 == 1:
            c1 = 4*J*h**2*(n-1) + 4*J**2*h*(n-1)
            c2 = 4*J*h**2*(n-1) + 4*J**2*h*(n-2)
        else:
            c1 = 4*J*h**2*(n-1) + 4*J**2*h*(n-2)
            c2 = 4*J*h**2*(n-1) + 4*J**2*h*(n-1)

    if verbose: print(f'c1 (analy)={c1}, c2={c2}')
    analytic_error_bound = c1 * dt**3 / 12 + c2 * dt**3 / 24
    return analytic_error_bound, c1, c2

def analytic_loose_commutator_bound_xyz(n, J, h, dt, pbc=False, verbose=False):
    if pbc:
        c1 = 16*J**2*h*(n) + 4*J**2*h*(n)
        c2 = 8*(n)*J**2*h
    else:
        c1 = 8*J*h**2*2*(n-1) 
        c2 =  8*J**2*h*(2*(n-1)-2) + 4*J**2*h*(2)

    if verbose: print(f'c1 (analy)={c1}, c2={c2}')
    analytic_error_bound = c1 * dt**3 / 12 + c2 * dt**3 / 24
    return analytic_error_bound, c1, c2

def analy_st_loose_bound(r, n, J, h, t, ob_type='single', group='parity'):
    if group == 'parity':
        return 2 * analytic_loose_commutator_bound_parity(n, J, h, t/r)[0] * r
    elif group == 'xyz':
        return 2 * analytic_loose_commutator_bound_xyz(n, J, h, t/r)[0] * r
    else:
        raise ValueError(f'group={group} not recognized')

def analy_st_bound(r, n, J, h, t, ob_type='single'):
    if ob_type == 'single':
        return 2 * analytic_loose_commutator_bound(n, J, h, t/r) * r
    elif ob_type == 'multi':
        # return 2 * analytic_loose_commutator_bound(n, J, h, t/r) * r  * n
        return 2 * analytic_loose_commutator_bound(n, J, h, t/r) * r 
    else:
        raise ValueError('ob_type should be either single or multi')

def analy_lc_bound(r, n, J, h, t, ob_type='single', verbose=False):
    err_bound = 0
    for i in range(1, r+1):
        if ob_type == 'single':
            n_lc = min(i*2, n)
            err_bound += 2 * analytic_loose_commutator_bound(n_lc, J, h, t/r, verbose) 
        elif ob_type == 'multi':   
            for j in range(0, n):
                n_lc = min(min(n-j, i*2) + min(j, 2*i), n)
                # err_bound += 2 * analytic_loose_commutator_bound(n_lc, J, h, t/r, verbose) 
                err_bound += 2 * analytic_loose_commutator_bound(n_lc, J, h, t/r, verbose) / n
        else:
            raise ValueError('ob_type should be either single or multi')

    return err_bound

def analytic_loose_commutator_bound(n, J, h, dt, pbc=False, verbose=False):
    if pbc:
        c1 = 16*J**2*h*(n) + 4*J**2*h*(n)
        c2 = 8*(n)*J**2*h
    else:
        # c1 = 16*J**2*h*(n-1) + 4*J**2*h*(n-2)
        # c2 = 8*(n-1)*J**2*h
        if n % 2 == 1:
            c1 = 4*J*h**2*(n-1) + 4*J**2*h*(n-1)
            c2 = 4*J*h**2*(n-1) + 4*J**2*h*(n-2)
        else:
            c1 = 4*J*h**2*(n-1) + 4*J**2*h*(n-2)
            c2 = 4*J*h**2*(n-1) + 4*J**2*h*(n-1)


    if verbose: print(f'c1 (analy)={c1}, c2={c2}')
    analytic_error_bound = c1 * dt**3 / 12 + c2 * dt**3 / 24
    return analytic_error_bound

from spin_ham import *

# def relaxed_commutator_bound(n, J, h, dt, verbose=False):
#     hnn = Nearest_Neighbour_1d(n, Jx=J, Jy=J, Jz=J, hx=h, hy=0, hz=0, pbc=False, verbose=False)
#     h_list = hnn.ham_par
def relaxed_commutator_bound(n, cmm_data, dt, verbose=False):
    relaxed_error_bound = cmm_data['c1'][n] * dt**3 / 12 + cmm_data['c2'][n] * dt**3 / 24
    return relaxed_error_bound

def relaxed_st_bound(r, n, cmm_data, t, ob_type='singl'):
    if ob_type == 'singl':
        return 2 * relaxed_commutator_bound(n, cmm_data, t/r) * r
    elif ob_type == 'multi':
        # return 2 * analytic_loose_commutator_bound(n, J, h, t/r) * r  * n
        return 2 * relaxed_commutator_bound(n, cmm_data, t/r) * r 
    else:
        raise ValueError('ob_type should be either single or multi')

def relaxed_lc_bound(r, n, cmm_data, t, ob_type='singl', verbose=False):
    err_bound = 0
    for i in range(1, r+1):
        if ob_type == 'singl':
            n_lc = min(i*2, n)
            err_bound += 2 * relaxed_commutator_bound(n_lc, cmm_data, t/r, verbose) 
        elif ob_type == 'multi':   
            for j in range(0, n):
                n_lc = min(min(n-j, i*2) + min(j, 2*i), n)
                # err_bound += 2 * analytic_loose_commutator_bound(n_lc, J, h, t/r, verbose) 
                err_bound += 2 * relaxed_commutator_bound(n_lc, cmm_data, t/r, verbose) / n
        else:
            raise ValueError('ob_type should be either single or multi')

    return err_bound