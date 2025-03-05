# -*- coding: utf-8 -*-
#
#    Copyright (C) 2025 Anton Akerman (a.a@b.com)
#                       Enis Chenchene (enis.chenchene@univie.ac.at)
#                       Pontus Giselsson (p.g@un.com)
#                       Emanuele Naldi (emanuele.naldi@unige.it)
#
#    This file is part of the example code repository for the paper:
#
#      A. Akerman, E. Chenchene, P. Giselsson, E. Naldi.
#      Characterization of Nonexpansive Forward-Backward-type Algorithms with
#      Minimal Memory Requirements,
#      2025. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
This file contains useful tools to reproduce the numerical experiments in:

A. Akerman, E. Chenchene, P. Giselsson, E. Naldi.
Characterization of Nonexpansive Forward-Backward-type Algorithms with
Minimal Memory Requirements,
2025. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.

"""

import numpy as np
import cvxpy as cp


def prox_norm_ell_2(tau, w):

    norm = np.linalg.norm(w)

    if norm <= 1e-9:
        return w
    else:
        return max(0, 1 - tau / norm) * w


def prox_norm_ell_1(tau, w):

    return np.sign(w) * np.maximum(np.abs(w) - tau, 0)


def prox_norm_ell_2_tilted(tau, w, tilt):
    '''
    computes the proximity operator of |w - tilt|_2
    '''

    return tilt + prox_norm_ell_2(tau, w - tilt)


def prox_norm_ell_1_tilted(tau, w, tilt):
    '''
    computes the proximity operator of |w - tilt|_1
    '''

    return tilt + prox_norm_ell_1(tau, w - tilt)


def create_prox_function_ell_2(s):

    return lambda tau, w: prox_norm_ell_2_tilted(tau, w, s)


def create_prox_function_ell_1(s):

    return lambda tau, w: prox_norm_ell_1_tilted(tau, w, s)


def create_Proxs(Anchors):

    return [create_prox_function_ell_2(anchor) for anchor in Anchors.T]


def create_grad_function(delta, A_j, y_j):

    return lambda x: A_j.T @ np.array(dhub(delta, A_j @ x - y_j), ndmin=1)


def create_grad_function_hub_flat(delta_1, delta_2, A_j, y_j):

    return lambda x: A_j.T @ np.array(dhub_flat(delta_1, delta_2, A_j @ x - y_j), ndmin=1)


def create_Grads(delta, f, A, y):
    '''
    ####################### no longer used
    '''
    m, dim = A.shape

    # splitting the forward terms into different chunks of balanced size
    chunks_indeces = np.array_split(np.arange(m), f)

    Grads = []
    betas = []

    for j in range(f):
        A_j = A[chunks_indeces[j], :]
        y_j = y[chunks_indeces[j]]

        Grads.append(create_grad_function(delta, A_j, y_j))
        betas.append(np.linalg.norm(A_j.T @ A_j, 2))

    return Grads, betas


def create_Grads_hub_flat(delta_1, delta_2, f, A, y):

    m, dim = A.shape

    # splitting the forward terms into different chunks of balanced size
    chunks_indeces = np.array_split(np.arange(m), f)

    Grads = []
    betas = []

    for j in range(f):
        A_j = A[chunks_indeces[j], :]
        y_j = y[chunks_indeces[j]]

        Grads.append(create_grad_function_hub_flat(delta_1, delta_2, A_j, y_j))
        betas.append(np.linalg.norm(A_j.T @ A_j, 2))

    return Grads, betas


def create_N_and_K(F, f, b, Range_N, Range_K):
    '''
    NOTE: F = [0, x, y, ..., z, b] with length b. F[i] denotes the forward
    terms (possibly -- it also depends on N) activated while evaluating the
    ith backward term.

    N and K are sampled unformly on the interval [Range[0], Range[1]]
    '''

    N = np.random.uniform(Range_N[0], Range_N[1], (b, f))
    K = np.random.uniform(Range_K[0], Range_K[1], (f, b))

    for i in range(b):
        N[i, F[i]:] = 0 * N[i, F[i]:]
        K[:F[i], i] = 0 * K[:F[i], i]

    N = N / np.sum(N, axis=0)[np.newaxis, :]
    K = K / np.sum(K, axis=1)[:, np.newaxis]

    return N, K


def create_N_and_K_DFB(f, b):
    '''
    Implements the N and K in Example REF in the paper.
    '''

    maximum_forward_terms = int(b * (b - 1) / 2)
    if f > maximum_forward_terms:
        raise Exception("In Example REF, f must be lower than b * (b - 1) / 2")

    if f < maximum_forward_terms:
        # deleting some rows of N and K to match the number of forward terms
        deleting_indeces = np.random.choice(maximum_forward_terms,
                                            size=maximum_forward_terms - f,
                                            replace=False)

    N = np.zeros((b, maximum_forward_terms))
    K = np.zeros((maximum_forward_terms, b))

    i_minus = 0
    for i in range(b):
        i_plus = i_minus + i
        ones_indeces = np.arange(i_minus, i_plus)
        i_minus = i_plus

        N[i, ones_indeces] = N[i, ones_indeces] + 1

        if i >= 1:
            K[ones_indeces, :len(ones_indeces)] = np.eye(i)

    # deleting excessive rows and columns
    if f < maximum_forward_terms:
        N = np.delete(N, deleting_indeces, axis=1)
        K = np.delete(K, deleting_indeces, axis=0)

    # retrieving F
    F = [0]

    for i in range(1, b):
        index = 0
        for j in range(f):
            if N[i, j] != 0:
                index = j + 1

        F.append(max(index, F[-1]))

    return N, K, F

def create_N_and_K_optimized(F, f, b, beta_diag):
    '''
    Chooses N and K to minimize 2-norm sqrt(beta_diag) * (H.T - K)

    Parameters
    ----------
    F : (m,n)-nondecreasing vector defining order
    f : number of forward evaluations
    b : number of backward evaluations
    beta_diag : diagonal matrix of beta-values
    '''

    K = cp.Variable((f, b))
    N = cp.Variable((b, f))
    constraints =[K @ np.ones((b, 1)) == np.ones((f, 1)),
                  N.T @ np.ones((b, 1)) == np.ones((f, 1))]

    for i in range(b):
        for j in range(f):
            if j >= F[i]:
                constraints.append(N[i,j] == 0)
            else:
                constraints.append(K[j,i] == 0)

    prob = cp.Problem(cp.Minimize(cp.norm2(np.sqrt(beta_diag) @ (N.T - K))),
                      constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)

    return N.value, K.value


def create_N_and_K_ACL24(f, b):
    '''
    Implements the N and K as in

    2024. Artacho, Campoy, Lopez-Pastor.
    '''

    maximum_forward_terms = b - 1
    if f > maximum_forward_terms:
        raise Exception("In ACL24, f must be equal or lower than b - 1")

    if f < maximum_forward_terms:
        # deleting some rows of N and K to match the number of forward terms
        deleting_indeces = np.random.choice(maximum_forward_terms,
                                            size=maximum_forward_terms - f,
                                            replace=False)

    N = np.zeros((b, maximum_forward_terms))
    K = np.zeros((maximum_forward_terms, b))

    N[1:, :] = np.eye(b - 1)

    # defining K
    for i in range(b - 1):
        K_i = np.zeros(i + 1)
        K_i[np.random.randint(i + 1)] = 1
        K[i, :(i + 1)] = K_i

    # deleting excessive rows and columns
    if f < maximum_forward_terms:
        N = np.delete(N, deleting_indeces, axis=1)
        K = np.delete(K, deleting_indeces, axis=0)

    # retrieving F
    F = [0]

    for i in range(1, b):
        index = 0
        for j in range(f):
            if N[i, j] != 0:
                index = j + 1

        F.append(max(index, F[-1]))

    return N, K, F


def create_N_and_K_AMTT23(f, b):
    '''
    Implements the N and K as in

    2023. Artacho, Malitsky, Tam, Torregrosa-Belén
    '''

    maximum_forward_terms = b - 1
    if f > maximum_forward_terms:
        raise Exception("In Example AMTT23, f must be equal or lower than b - 1")

    if f < maximum_forward_terms:
        # deleting some rows of N and K to match the number of forward terms
        deleting_indeces = np.random.choice(maximum_forward_terms,
                                            size=maximum_forward_terms - f,
                                            replace=False)

    N = np.zeros((b, maximum_forward_terms))
    K = np.zeros((maximum_forward_terms, b))

    N[1:, :] = np.eye(b - 1)
    K[:, :-1] = np.eye(b - 1)

    # deleting excessive rows and columns
    if f < maximum_forward_terms:
        N = np.delete(N, deleting_indeces, axis=1)
        K = np.delete(K, deleting_indeces, axis=0)

    # retrieving F
    F = [0]

    for i in range(1, b):
        index = 0
        for j in range(f):
            if N[i, j] != 0:
                index = j + 1

        F.append(max(index, F[-1]))

    return N, K, F


def hub(delta, z):

    # force it as array
    z = np.array(z, ndmin=1)
    norms = np.abs(z)

    out = np.copy(norms)

    out[norms <= delta] = 0.5 * out[norms <= delta] ** 2
    out[norms > delta] =  delta * (out[norms > delta] - 0.5 * delta)

    return np.sum(out)


def dhub(delta, z):

    # force it as array
    z = np.array(z, ndmin=1)
    norms = np.abs(z)

    out = np.copy(z)

    out[norms > delta] = delta * np.sign(z[norms > delta])

    if len(out) == 1:
        return out[0]
    else:
        return out


def hub_flat(delta_1, delta_2, z):
    '''
    Computes the function

    hub_flat(z) := sum_i h(z_i)

    where h: R to R is defined for all z_i in R by:

    h(z_i) :=
    0 if |z_i| <= delta_1
    0.5 * (|z_i| - delta_1) ** 2 if delta_1 <= |z_i| <= delta_2
    (delta_2 - delta_1) * z_i - 0.5 * (delta_2 ** 2 - delta_1 ** 2)
    '''

    # force it as array
    z = np.array(z, ndmin=1)
    norms = np.abs(z)

    out = np.copy(norms)

    middle_range = (delta_1 <= norms) * (norms <= delta_2)
    out[middle_range] = 0.5 * (out[middle_range] - delta_1) ** 2
    out[norms < delta_1] = 0 * out[norms < delta_1]
    out[norms > delta_2] = (delta_2 - delta_1) * out[norms > delta_2] \
        - 0.5 * (delta_2 ** 2 - delta_1 ** 2)

    return np.sum(out)


def dhub_flat(delta_1, delta_2, z):
    '''
    Computes gradient of the function dhub_flat defined above.

    '''

    # force it as array
    z = np.array(z, ndmin=1)
    norms = np.abs(z)

    out = np.copy(z)
    middle_range = (delta_1 <= norms) * (norms <= delta_2)

    out[middle_range] = np.sign(out[middle_range]) * (norms[middle_range] - delta_1)
    out[norms < delta_1] = 0 * out[norms < delta_1]
    out[norms > delta_2] = (delta_2 - delta_1) * np.sign(out[norms > delta_2])

    if len(out) == 1:
        return out[0]
    else:
        return out


def fobj_exp2(x, A, y, Anchors, delta):
    '''
    #################### No longer used
    '''

    fidelity = hub(delta, A @ x - y)
    non_smooth = np.sum(np.linalg.norm(x[:, np.newaxis] - Anchors, axis=0))

    return fidelity + non_smooth


def fobj_exp1(x, A, y, Anchors, delta_1, delta_2):
    '''
    #################### No longer used
    '''

    fidelity = hub_flat(delta_1, delta_2, A @ x - y)
    non_smooth = np.sum(np.linalg.norm(x[:, np.newaxis] - Anchors, axis=0))

    return fidelity + non_smooth
