# -*- coding: utf-8 -*-
#
#    Copyright (C) 2025 Anton Akerman (anton.akerman@control.lth.se)
#                       Enis Chenchene (enis.chenchene@univie.ac.at)
#                       Pontus Giselsson (pontusg@control.lth.se)
#                       Emanuele Naldi (emanuele.naldi@unige.it)
#
#    This file is part of the example code repository for the paper:
#
#      A. Akerman, E. Chenchene, P. Giselsson, E. Naldi.
#      Splitting the Forward-Backward Algorithm: A Full Characterization.
#      2025. DOI: 10.48550/arXiv.2504.10999.
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
Splitting the Forward-Backward Algorithm: A Full Characterization.
2025. DOI: 10.48550/arXiv.2504.10999.

For any comment, please contact: enis.chenchene@gmail.com
"""

import numpy as np
import structures as st
import operators as op
import networkx as nx


def aGFB(Proxs, Grads, betas, w_init, maxit, tau, Model,
         Compute_Dist_to_Sol=False):
    '''
    Implements the adapted graph forward-backward (aGFB) Method, introduced in
    Section 5.
    '''

    # retrieving information
    f = len(Grads)
    b = len(Proxs)

    # storage
    Vars = np.zeros(maxit)
    Objs = np.zeros(maxit)
    Dist = np.zeros(maxit)

    N, K, F = st.create_N_and_K_aGFB(f, b)
    Lap = b * np.eye(b) - np.ones(b)

    sLap = 0 * Lap
    J = op.FBO(tau, Proxs, Grads, Model.dim, betas, Lap, sLap, N, K, F)

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


def SFB_plus(Proxs, Grads, betas, w_init, maxit, tau, Model,
             Compute_Dist_to_Sol=False):
    '''
    Implements the Split-Forward-Backward+ (DFB) Method, introduced in
    Section 6, with H and K matrices minimizing |W|_2.
    '''

    # retrieving information
    f = len(Grads)
    b = len(Proxs)

    # storage
    Vars = np.zeros(maxit)
    Objs = np.zeros(maxit)
    Dist = np.zeros(maxit)

    F = [0] + [f // (b - 1) * i for i in range(1, b - 1)] + [f]
    N, K = st.create_N_and_K_optimized(F, f, b, np.diag(betas))

    Lap = b * np.eye(b) - np.ones(b)

    sLap = 0 * Lap
    J = op.FBO(tau, Proxs, Grads, Model.dim, betas, Lap, sLap, N, K, F)

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
    J = op.FBO(tau, Proxs, Grads, Model.dim, betas, Lap, sLap, N, K, F)

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

    J = op.FBO(tau, Proxs, Grads, Model.dim, betas, Lap, sLap, N, K, F)

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

    J = op.FBO(tau, Proxs, Grads, Model.dim, betas, Lap, sLap, N, K, F)

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

    if type(Range_N) is not int and Range_N is not None:
        N, K = st.create_N_and_K(F, f, b, Range_N, Range_K)

    Lap = b * np.eye(b) - np.ones(b)

    sLap = 0 * Lap
    J = op.FBO(tau, Proxs, Grads, Model.dim, betas, Lap, sLap, N, K, F)

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


def General_Instance(Proxs, Grads, betas, F, w_init, maxit, tau, Model, Lap,
                     sLap, N, K):
    '''
    Implements Algorithm 1 with general Lap, sLap, N and K.
    '''

    # storage
    Vars = np.zeros(maxit)
    Objs = np.zeros(maxit)

    J = op.FBO(tau, Proxs, Grads, Model.dim, betas, Lap, sLap, N, K, F)

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
