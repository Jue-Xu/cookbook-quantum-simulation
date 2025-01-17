# Various utility functions used by the scripts
import itertools
import math
import random, sys
from cmath import cos, exp, pi, sin, sqrt
from numpy.linalg import matrix_power
from scipy.linalg import expm
from numpy import log
import numpy as np
np.set_printoptions(precision=6)
FLOATING_POINT_PRECISION = 1e-10
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse.linalg as ssla

import multiprocessing

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.colors import ListedColormap
import colorsys
# colors = mpl.cycler(color=["c", "m", "y", "r", "g", "b", "k"]) 
# color_cycle = ["#4DBBD5FF", "#E64B35FF", "#00A087FF", "#70699e", "#F39B7FFF", "#3C5488FF", "#7E6148FF","#DC0000FF",  "#91D1C2FF", "#B09C85FF", "#923a3a", "#8491B4FF"]
# color_cycle = ["#E9002D", "#00B000", "#009ADE", "#FFAA00", "#6E005F"]
# color_cycle = ["#E64B35FF", "#47B167FF", "#0A75C7FF", "#F39B7FFF", "#70699eFF", "#4DBBD5FF", "#FFAA00FF"]
color_cycle = ["#b5423dFF", "#405977FF", "#616c3aFF", "#e3a13aFF", "#7a2c29FF", "#253a6aFF", "#8b9951FF"]
color_cycle = [color[:-2] + "FF" for color in color_cycle]
# Function to lighten a color
def lighten_color(color, amount=0.3):
    # Convert color from hexadecimal to RGB
    r, g, b, a = tuple(int(color[i:i+2], 16) for i in (1, 3, 5, 7))
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    # Lighten the luminance component
    l = min(1, l + amount)
    # Convert HLS back to RGB
    r, g, b = tuple(round(c * 255) for c in colorsys.hls_to_rgb(h, l, s))
    # Convert RGB back to hexadecimal
    new_color = f"#{r:02x}{g:02x}{b:02x}{a:02x}"
    return new_color

color_cycle_light = [lighten_color(color, 0.3) for color in color_cycle]
# color_cycle_light = [color[:-2] + "60" for color in color_cycle]
colors = mpl.cycler(mfc=color_cycle_light, color=color_cycle, markeredgecolor=color_cycle)

# import scienceplots
# plt.style.use(['science', 'nature'])

mpl.rc('axes', prop_cycle=colors)
# mpl.rcParams['axes.prop_cycle'] = colors
# mpl.rcParams['lines.markeredgecolor'] = 'C'
# mpl.use('ps')
# mpl.use("pgf")
# mpl.rc("pgf", texsystem="pdflatex", preamble=r"\usepackage{color}")

# mpl.rc('text',usetex=True)
# mpl.rc('text.latex', preamble='\\usepackage{color}')
# mpl.rc('axes', grid=True, edgecolor='k', prop_cycle=colors)
plt.rcParams['font.family'] = 'Helvetica'
# plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['legend.frameon'] = False
# plt.rcParams['lines.markeredgecolor'] = 'k'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markeredgewidth'] = 1.0
mpl.rcParams['figure.dpi'] = 100  # 130

SMALL_SIZE = 14
MEDIUM_SIZE = 15  #default 10
LARGE_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE+2)  # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=LARGE_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=LARGE_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

def jax_matrix_exponential(matrix):
    # return jsl.expm( matrix)
    return ssla.expm(matrix)
jax_matrix_exponential = jax.jit(jax.vmap(jax_matrix_exponential))

def sparse_multi_dot(sparse_matrices):
    '''
    计算一个列表中所有矩阵的乘积
    '''
    product = sparse_matrices[0]
    for matrix in sparse_matrices[1:]:
        product = product.dot(matrix)
    return product
    # return product.toarray()

vectorized_sparse_expm = jax.vmap(ssla.expm)

def mpi_sparse_expm(list_herms, t, r):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    list_unitaries = pool.map(ssla.expm, -1j * t / r * np.array(list_herms))
    # Close the pool of workers
    pool.close()
    pool.join()

    return list_unitaries

def sparse_trotter_error(list_herm: list, r: int, t: int) -> float:
    print('-------sparse_trotter_error--------')
    exact_U = ssla.expm(-1j * t * sum(list_herm))
    # list_U = jax_matrix_exponential(jnp.array(-1j * t / (2*r) * np.array(list_herm)))
    # list_U = vectorized_sparse_expm(-1j * t / (2*r) * np.array(list_herm))
    # list_herm_scaled = np.array([-1j * t / (2*r) * herm for herm in list_herm])
    # list_U = ssla.expm(list_herm_scaled) 
    # list_U = [ssla.expm(-1j * t / (2*r) * herm) for herm in list_herm]
    list_U = mpi_sparse_expm(list_herm, t, 2*r)
    # list_U = jax_matrix_exponential(jnp.array([-1j * t / (2*r) * herm.toarray() for herm in np.array(list_herm)]))
    list_U2 = [U**2 for U in list_U]
    # trotter_error_list = op_error(exact_U, matrix_power(sparse_multi_dot(list_U2), r))
    trotter_error_list = op_error(exact_U, sparse_multi_dot(list_U2)**r)
    # trotter_error_list = op_error(exact_U, np.linalg.matrix_power(np.linalg.multi_dot(np.array(list_U2)), r))
    # second-order trotter
    trotter_error_list_2nd = op_error(exact_U, (sparse_multi_dot(list_U) @ sparse_multi_dot(list_U[::-1]))**r)
    # trotter_error_list_2nd = op_error(exact_U, np.linalg.matrix_power(np.linalg.multi_dot(np.array(list_U)) @ np.linalg.multi_dot(np.array(list_U[::-1])), r))
    
    return [trotter_error_list, trotter_error_list_2nd]

# matrix product of a list of matrices
def unitary_matrix_product(list_herm_matrices, t=1):
    ''' 
    matrix product of a list of unitary matrices exp(itH)
    input: 
        list_herm_matrices: a list of Hermitian matrices
        t: time
    return: the product of the corresponding matrices
    '''
    product = expm(-1j * t * list_herm_matrices[0])
    for i in range(1, len(list_herm_matrices)):
        product = product @ expm(-1j * t * list_herm_matrices[i])

    return product

def matrix_product(list_U, t=1):
    # product = matrix_power(list_U[0], t)
    # for i in range(1, len(list_U)):
    #     product = matrix_power(list_U[i], t) @ product
    #     # product = product @ matrix_power(list_U[i], t)
    product = np.linalg.multi_dot([matrix_power(U, t) for U in list_U])
    return product

def commutator(A, B):
    return A @ B - B @ A

# def anticommutator(A, B, to_sparse=False):
def anticommutator(A, B):
    return A @ B + B @ A

def second_order_trotter(list_herm_matrices, t=1):
    forward_order_product = unitary_matrix_product(list_herm_matrices, t/2) 
    reverse_order_product = unitary_matrix_product(list_herm_matrices[::-1], t/2)

    return forward_order_product @ reverse_order_product

def pf_U(list_U, order, t=1):
    # print('order: ', order)
    if order == 1:
        return matrix_product(list_U, t)
    elif order == 2:
        forward_order_product = matrix_product(list_U, t/2) 
        reverse_order_product = matrix_product(list_U[::-1], t/2)
        return forward_order_product @ reverse_order_product
    elif order > 0 and order != 1 and order != 2 and order % 2 == 0:
        p = 1 / (4 - 4**(1/(order-1)))
        # print('p: ', p)
        return matrix_power(pf_U(list_U, order-2, p*t), 2) @ pf_U(list_U, order-2, (1-4*p)*t) @ matrix_power(pf_U(list_U, order-2, p*t), 2)
    else:
        raise ValueError('k is not defined')

def pf(list_herm, order, t):
    # print('order: ', order)
    if order == 1:
        return unitary_matrix_product(list_herm, t)
    elif order == 2:
        forward_order_product = unitary_matrix_product(list_herm, t/2) 
        reverse_order_product = unitary_matrix_product(list_herm[::-1], t/2)
        return forward_order_product @ reverse_order_product
        # return second_order_trotter(list_herm, t)
    elif order > 0 and order!= 1 and order != 2 and order % 2 == 0:
        p = 1 / (4 - 4**(1/(order-1)))
        # print('p: ', p)
        return matrix_power(pf(list_herm, order-2, p*t), 2) @ pf(list_herm, order-2, (1-4*p)*t) @ matrix_power(pf(list_herm, order-2, p*t), 2)
    else:
        raise ValueError('k is not defined')

def op_error(exact, approx, norm='spectral'):
    ''' 
    Frobenius norm of the difference between the exact and approximated operator
    input:
        exact: exact operator
        approx: approximated operator
    return: error of the operator
    '''
    if norm == 'fro':
        return jnp.linalg.norm(exact - approx)
    elif norm == 'spectral':
        # if the input is in csr_matrix format
        if isinstance(exact, csc_matrix) and isinstance(approx, csc_matrix):
            return jnp.linalg.norm(jnp.array(exact.toarray() - approx.toarray()), ord=2)
        else:
            return jnp.linalg.norm(exact - approx, ord=2)
    else:
        raise ValueError('norm is not defined')
    # return np.linalg.norm(exact - approx)/len(exact)

# evaluate trotter error for different number of trotter steps
def trotter_error(list_herm, r_list, t, norm='spectral', n_perm=50, verbose=False):
    ''' 
    evaluate trotter error for different number of trotter steps
    input: 
        list_herm: a list of Hermitian matrices
        r_list: number of trotter steps
    return: trotter error
    '''
    exact_U = expm(-1j * t * sum(list_herm))
    list_U = [expm(-1j * t / (2*r_list[-1]) * herm) for herm in list_herm]
    if len(list_U) >= 5:
        print('number of terms: ', len(list_U))
        perm_list = [list_U] 
        seed_value = random.randrange(sys.maxsize)
        random.seed(seed_value)  
        # randomly select 5 permutations from perm_list
        for _ in range(n_perm-1):
            # random.shuffle(list_U) 
            # perm_list.append(list_U[:])
            perm_list.append(random.sample(list_U, len(list_U)))
        # perm_list = random.sample(perm_list, 50) 
        print('# randomly selected perm: ', len(perm_list))
    else:
        # generate a list of permutation of the order of the matrices
        perm_list = list(itertools.permutations(list_U))
        # print('perm_list', perm_list)
        print('# all perm: ', len(perm_list))
    # perm_list = list(itertools.permutations(list_herm))[:5]
    # for r in r_list:
    # first-order trotter
    trotter_error_list = [op_error(matrix_power(matrix_product(perm, int(2*r_list[-1]/r)), r), exact_U, norm) for r in r_list for perm in perm_list]
    # trotter_error_list = [op_error(matrix_power(unitary_matrix_product(perm, t=t/r), r), exact_U, norm) for r in r_list for perm in perm_list]
    # second-order trotter
    trotter_error_list_2nd = [op_error(matrix_power(matrix_product(perm, int(r_list[-1]/r)) @ matrix_product(perm[::-1], int(r_list[-1]/r)), r), exact_U, norm) for r in r_list for perm in perm_list]
    err_1st_reshaped = np.array(trotter_error_list).reshape(len(r_list), len(perm_list))
    err_2nd_reshaped = np.array(trotter_error_list_2nd).reshape(len(r_list), len(perm_list))

    return err_1st_reshaped , err_2nd_reshaped

def search_r_for_error(r_start, r_end, epsilon, t, list_herm, k, norm='spectral', verbose=False):
    tol = r_end - r_start
    exact_U = expm(-1j * t * sum(list_herm))
    # binary search from r_start to r_end
    while tol > 2:
        r = int((r_start + r_end) / 2)
        err = op_error(matrix_power(pf(list_herm, k, t=t/r), r), exact_U, norm)
        # if k == 1:
        #     err = op_error(matrix_power(unitary_matrix_product(list_herm, t=t/r), r), exact_U, norm)
        # elif k == 2:
        #     err = op_error(matrix_power(second_order_trotter(list_herm, t=t/r), r), exact_U, norm)
        # elif k != 2 and k > 1 and k % 2 == 0:
        #     err = op_error(matrix_power(high_order_trotter(list_herm, k, t=t/r), r), exact_U, norm)
        # else:
        #     raise ValueError('k is not defined')

        if err > epsilon:
            r_start = r
        else:
            r_end = r
        tol = abs(r_end - r_start)
    if verbose: print('err: ', err)
    return r

def plot_trotter_error_vs_r(epsilon, t, ham_group, r_list, perm_label, markers, plot=True, locate=True):
    trotter_error_list, trotter_error_list_2nd = trotter_error(ham_group, r_list, t)
    # print('trotter_error_list: \n', trotter_error_list)
    # for index, trotter_error in enumerate(trotter_error_list):
    #     plt.plot(r_list, trotter_error, '^-', label='ordering: '+perm_label[index])
    if plot:
        for i in range(len(trotter_error_list[0])):
            plt.plot(r_list, trotter_error_list[:,i], markers[i], markeredgecolor='black', label= perm_label[i] + ' (1st)')

        for i in range(len(trotter_error_list_2nd[0])):
            plt.plot(r_list, trotter_error_list_2nd[:,i], markers[i], markeredgecolor='black', label=perm_label[i] + ' (2nd)')

    if locate:
        epsilon_list = [epsilon] * len(trotter_error_list[:, 0])
        idx_1st_0 = np.argwhere(np.diff(np.sign(epsilon_list - trotter_error_list[:,0])))
        idx_1st_1 = np.argwhere(np.diff(np.sign(epsilon_list - trotter_error_list[:,1])))
        idx_2nd_0 = np.argwhere(np.diff(np.sign(epsilon_list - trotter_error_list_2nd[:,0])))
        idx_2nd_1 = np.argwhere(np.diff(np.sign(epsilon_list - trotter_error_list_2nd[:,1])))
        intersect_indices = [ r_list[index] for index in np.array([idx_1st_0, idx_1st_1, idx_2nd_0, idx_2nd_1]).flatten() ]
        print('intersect_indices: ',intersect_indices)

        return intersect_indices

def data_plot(x, y, marker, label, alpha=1, linewidth=1, loglog=True, markeredgecolor='black'):
    if loglog:
        plt.loglog(x, y, marker, label=label, linewidth=linewidth, markeredgecolor=markeredgecolor, markeredgewidth=0.5, alpha=alpha)
    else:
        plt.plot(x, y, marker, label=label, linewidth=linewidth, markeredgecolor=markeredgecolor, markeredgewidth=0.5, alpha=alpha)

from scipy.optimize import curve_fit
from math import ceil, floor, log, exp

def linear_loglog_fit(x, y, verbose=False):
    # Define the linear function
    def linear_func(x, a, b):
        return a * x + b

    log_x = np.array([log(n) for n in x])
    log_y = np.array([log(cost) for cost in y])
    # Fit the linear function to the data
    params, covariance = curve_fit(linear_func, log_x, log_y)
    # Extract the parameters
    a, b = params
    # Predict y values
    y_pred = linear_func(log_x, a, b)
    # Print the parameters
    if verbose: print('Slope (a):', a, 'Intercept (b):', b)
    exp_y_pred = [exp(cost) for cost in y_pred]

    return exp_y_pred, a, b

def plot_fit(ax, x, y, var='t', offset=1.07, verbose=True):
    y_pred_em, a_em, b_em = linear_loglog_fit(x, y)
    if verbose: print(f'a_em: {a_em}; b_em: {b_em}')
    text_a_em = "{:.2f}".format(round(abs(a_em), 4))
    y_pred_em = [exp(cost) for cost in a_em*np.array([log(n) for n in x]) + b_em]
    ax.plot(x, y_pred_em, 'k--', linewidth=1)
    ax.annotate(r'$O(%s^{%s})$' % (var, text_a_em), xy=(x[-1], np.real(y_pred_em)[-1]), xytext=(x[-1]*offset, np.real(y_pred_em)[-1]))