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
This file contains an implementation of several methods used in:

A. Akerman, E. Chenchene, P. Giselsson, E. Naldi.
Characterization of Nonexpansive Forward-Backward-type Algorithms with
Minimal Memory Requirements,
2025. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.

"""

import numpy as np
import structures as st
import operators as op
import networkx as nx


def DFB(Proxs, Grads, betas, w_init, maxit, tau, Model,
        Compute_Dist_to_Sol=False):
    '''
    Implements the Distributed Forward Backward (DFB) Method, introduced in
    Section REF.
    '''

    # retrieving information
    f = len(Grads)
    b = len(Proxs)

    # storage
    Vars = np.zeros(maxit)
    Objs = np.zeros(maxit)
    Dist = np.zeros(maxit)

    N, K, F = st.create_N_and_K_DFB(f, b)
    Lap =  b * np.eye(b) - np.ones(b)

    sLap =  0 * Lap
    J = op.genFBO(tau, Proxs, Grads, Model.dim, betas, Lap, sLap, N, K, F)

    w = np.copy(w_init)
    for k in range(maxit):

        # step
        w, x = J.apply(w)
        mean_x = np.mean(x, axis=0)

        # compute objective value
        obj = Model.objective(mean_x)
        Objs[k] = obj

        # computing variance
        var = np.sum((x - mean_x[np.newaxis, :]) ** 2)
        Vars[k] = var

        # computing distance to solution
        if Compute_Dist_to_Sol:
            Dist[k] = Model.distance_to_solution(x[1, :])

    if Compute_Dist_to_Sol:
        return Vars, Objs, Dist, x[1, :]

    else:
        return Vars, Objs


def F_heuristic(f, b, betas):
    """
    Heuristic for finding F, accounting for cocoercivity-constants beta.
    Tries to make sure sum is equal for the cocoercive operators
    between each pair of consecutive resolvents
    """
    F = [0]
    beta_sum = 0
    tot_beta = sum(betas)
    b_left = b - 1

    for i in range(f):
        if abs(beta_sum - tot_beta / b_left) <= \
            abs(beta_sum + betas[i] - tot_beta / b_left):
            F.append(i)
            beta_sum = betas[i]
        else:
            beta_sum += betas[i]
    F = F + [f] * (b - len(F))
    F[-1] = f

    return F


def DFB_optimized(Proxs, Grads, betas, w_init, maxit, tau, Model,
                  Compute_Dist_to_Sol=False, heuristic=False):
    '''
    Implements the Distributed Forward Backward (DFB) Method, introduced in
    Section REF, with H and K matrices minimizing |W|_2.
    Heuristic determines if heristic is used to generate F. If False, splits
    evenly.
    '''

    # retrieving information
    f = len(Grads)
    b = len(Proxs)

    # storage
    Vars = np.zeros(maxit)
    Objs = np.zeros(maxit)
    Dist = np.zeros(maxit)

    if heuristic:
        F = F_heuristic(f, b, betas)
    else:
        F = [0] + [f // (b - 1) * i for i in range(1, b - 1)] + [f]

    N, K = st.create_N_and_K_optimized(F, f, b, np.diag(betas))

    Lap =  b * np.eye(b) - np.ones(b)

    sLap = 0 * Lap
    J = op.genFBO(tau, Proxs, Grads, Model.dim, betas, Lap, sLap, N, K, F)

    w = np.copy(w_init)
    for k in range(maxit):

        # step
        w, x = J.apply(w)
        mean_x = np.mean(x, axis=0)

        # compute objective value
        obj = Model.objective(mean_x)
        Objs[k] = obj

        # computing variance
        var = np.sum((x - mean_x[np.newaxis, :]) ** 2)
        Vars[k] = var

        # computing distance to solution
        if Compute_Dist_to_Sol:
            Dist[k] = Model.distance_to_solution(x[1, :])

    if Compute_Dist_to_Sol:
        return Vars, Objs, Dist, x[1, :]

    else:
        return Vars, Objs


def ACL24(Proxs, Grads, betas, w_init, maxit, tau, Model,
          Compute_Dist_to_Sol=False):
    '''
    Implements the Forward Backward Method, introduced in
    2024. Artacho, Campoy, Lopez-Pastor.
    '''

    # retrieving information
    f = len(Grads)
    b = len(Proxs)

    # storage
    Vars = np.zeros(maxit)
    Objs = np.zeros(maxit)
    Dist = np.zeros(maxit)

    # forward an backward structures
    N, K, F = st.create_N_and_K_ACL24(f, b)
    Lap = b * np.eye(b) - np.ones(b)

    # state graph of Artacho is equal to
    sLap = tau * np.max(betas) / 4 * Lap - \
        1 / 4 * (N - K.T) @ np.diag(betas) @ (N.T - K)
    J = op.genFBO(tau, Proxs, Grads, Model.dim, betas, Lap, sLap, N, K, F)

    w = np.copy(w_init)
    for k in range(maxit):

        # step
        w, x = J.apply(w)
        mean_x = np.mean(x, axis=0)

        # compute objective value
        obj = Model.objective(mean_x)
        Objs[k] = obj

        # computing variance
        var = np.sum((x - mean_x[np.newaxis, :]) ** 2)
        Vars[k] = var

        # computing distance to solution
        if Compute_Dist_to_Sol:
            Dist[k] = Model.distance_to_solution(x[1, :])

    if Compute_Dist_to_Sol:
        return Vars, Objs, Dist, x[1, :]

    else:
        return Vars, Objs



def AMTT23(Proxs, Grads, betas, w_init, maxit, tau, Model,
           Compute_Dist_to_Sol=False):
    '''
    Implements the Forward Backward Method, introduced in
    2023. Artacho, Malitsky, Tam, Torregrosa-Belén
    '''

    # retrieving information
    f = len(Grads)
    b = len(Proxs)

    # storage
    Vars = np.zeros(maxit)
    Objs = np.zeros(maxit)
    Dist = np.zeros(maxit)

    N, K, F = st.create_N_and_K_AMTT23(f, b)

    # laplacian of the path graph
    Graph = nx.path_graph(b)
    Lap = nx.laplacian_matrix(Graph).toarray()

    # upper/state graph has the only edge (0, b)
    Graph = nx.Graph()
    Graph.add_nodes_from(range(b - 1))
    Graph.add_edge(0, b - 1)
    sLap = nx.laplacian_matrix(Graph).toarray()

    J = op.genFBO(tau, Proxs, Grads, Model.dim, betas, Lap, sLap, N, K, F)

    w = np.copy(w_init)
    for k in range(maxit):

        # step
        w, x = J.apply(w)
        mean_x = np.mean(x, axis=0)

        # compute objective value
        obj = Model.objective(mean_x)
        Objs[k] = obj

        # computing variance
        var = np.sum((x - mean_x[np.newaxis, :]) ** 2)
        Vars[k] = var

        # computing distance to solution
        if Compute_Dist_to_Sol:
            Dist[k] = Model.distance_to_solution(x[1, :])

    if Compute_Dist_to_Sol:
        return Vars, Objs, Dist, x[1, :]

    else:
        return Vars, Objs


def BCLN23(Proxs, Grads, betas, w_init, maxit, tau, Model,
           Compute_Dist_to_Sol=False):
    '''
    Implements the Sequential Forward Backward Method, introduced in
    2023. Bredies, Chenchene, Lorenz, Naldi
    '''

    # retrieving information
    f = len(Grads)
    b = len(Proxs)

    # storage
    Vars = np.zeros(maxit)
    Objs = np.zeros(maxit)
    Dist = np.zeros(maxit)

    N, K, F = st.create_N_and_K_AMTT23(f, b)

    # laplacian of the path graph
    Graph = nx.path_graph(b)
    Lap = nx.laplacian_matrix(Graph).toarray()

    # upper/state graph has the only edge (1, b)
    sLap = 0 * Lap

    J = op.genFBO(tau, Proxs, Grads, Model.dim, betas, Lap, sLap, N, K, F)

    w = np.copy(w_init)
    for k in range(maxit):

        # step
        w, x = J.apply(w)
        mean_x = np.mean(x, axis=0)

        # compute objective value
        obj = Model.objective(mean_x)
        Objs[k] = obj

        # computing variance
        var = np.sum((x - mean_x[np.newaxis, :]) ** 2)
        Vars[k] = var

        # computing distance to solution
        if Compute_Dist_to_Sol:
            Dist[k] = Model.distance_to_solution(x[1, :])

    if Compute_Dist_to_Sol:
        return Vars, Objs, Dist, x[1, :]

    else:
        return Vars, Objs


def Random_Instance(Proxs, Grads, betas, F, w_init,
                    maxit, tau, Model, Range_N=0, Range_K=0, N=0, K=0):
    '''
    Implements Algorithm 1 with random instances of N and K with elements
    sampled from Range_N and Range_K respectively and Laplacian = fully
    connected.

    '''

    # retrieving information
    f = len(Grads)
    b = len(Proxs)

    # storage
    Vars = np.zeros(maxit)
    Objs = np.zeros(maxit)

    if type(Range_N) != int and Range_N != None:
        N, K = st.create_N_and_K(F, f, b, Range_N, Range_K)

    Lap =  b * np.eye(b) - np.ones(b)

    sLap = 0 * Lap
    J = op.genFBO(tau, Proxs, Grads, Model.dim, betas, Lap, sLap, N, K, F)

    w = np.copy(w_init)
    for k in range(maxit):

        # step
        w, x = J.apply(w)
        mean_x = np.mean(x, axis=0)

        # compute objective value
        obj = Model.objective(mean_x)
        Objs[k] = obj

        # computing variance
        var = np.sum((x - mean_x[np.newaxis, :]) ** 2)
        Vars[k] = var

    return Vars, Objs
