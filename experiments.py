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
#      Splitting the Forward-Backward Algorithm: A Full Characterization.
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
This file contains the numerical experiments in:

A. Akerman, E. Chenchene, P. Giselsson, E. Naldi.
Splitting the Forward-Backward Algorithm: A Full Characterization.
2025. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.

"""

import numpy as np
import structures as st
import operators as op
import optimization as optim
import plots as show
from tqdm import tqdm
import portfolio_optimization as port_opt
import networkx as nx


def experiment_testing_M(maxit=100):
    '''
    In this experiment, we test the influence of the matrix M
    '''

    np.random.seed(0)

    # problem and algorithm's parameters
    dim = 2       # dimension of the problem
    m = 10        # dimension of matrix (note: f must be <= than m)
    b = 20        # number of backward steps
    f = 10        # number of forward terms
    delta_1 = 1   # parameter for huber function
    delta_2 = 2
    tau = 1       # step-size

    # generating sample
    block_corruped_size = 2
    # np.random.seed(4)
    A = 2 * (np.random.rand(m, dim) - 0.5)
    noise_columns = np.random.randint(block_corruped_size, m)
    A[noise_columns, :] = 5 * A[noise_columns, :]
    y = np.random.rand(m)

    # generating anchor points
    Anchors = np.random.normal(0, 5, size=(dim, b))

    # initializing optimization problem
    Model = op.Model_Test(dim, A, y, Anchors, delta_1, delta_2)

    # defining backward operators
    Proxs = st.create_Proxs(Anchors)

    # cases
    cases = 200

    # initialization
    w_init = np.zeros((b, dim))

    # defining N and K (randomly)
    Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
    F = np.sort(np.random.randint(1, f + 1, b - 2))
    F = np.hstack(([0], F, [f]))
    N, K = st.create_N_and_K(F, f, b, [0, 1], [0, 1])

    # storage
    Vars = np.zeros((maxit, cases))
    Objs = np.zeros((maxit, cases))
    Spec = np.zeros(cases)

    for cs in tqdm(range(cases)):

        if cs == 0:
            Lap = b * np.eye(b) - np.ones(b)
            norm = np.linalg.norm(Lap, 2)
            Lap = Lap / norm

        elif cs >= 1 and cs < 20:
            Lap = np.random.rand(b, b - 1)
            Lap = Lap - np.mean(Lap, axis=0)[np.newaxis, :]
            Lap = Lap @ Lap.T
            norm = np.linalg.norm(Lap, 2)
            Lap = Lap / norm

        else:
            # sampling random Laplacian
            G = nx.connected_watts_strogatz_graph(b, np.random.randint(2, b), p=.8)
            Lap = nx.laplacian_matrix(G)
            Lap = Lap.toarray()
            norm = np.linalg.norm(Lap, 2)
            Lap = Lap / norm

        sLap = 0 * Lap

        # running the two methods
        Vars[:, cs], Objs[:, cs] = \
            optim.General_Instance(Proxs, Grads, betas, F, w_init,
                                   maxit, tau, Model, Lap, sLap, N, K)

        Spec[cs] = np.linalg.eigh(Lap)[0][1]

    show.plot_experiment_Lap(Objs, Spec, cases, maxit)
    #return Objs, Spec, cases, maxit


def experiment_testing_P(maxit=100):
    '''
    In this experiment, we test the influence of the matrix P
    '''

    np.random.seed(0)

    # problem and algorithm's parameters
    dim = 2       # dimension of the problem
    m = 10        # dimension of matrix (note: f must be <= than m)
    b = 20        # number of backward steps
    f = 10        # number of forward terms
    delta_1 = 1   # parameter for huber function
    delta_2 = 2
    tau = 1       # step-size

    # generating sample
    block_corruped_size = 2
    # np.random.seed(4)
    A = 2 * (np.random.rand(m, dim) - 0.5)
    noise_columns = np.random.randint(block_corruped_size, m)
    A[noise_columns, :] = 5 * A[noise_columns, :]
    y = np.random.rand(m)

    # generating anchor points
    Anchors = np.random.normal(0, 5, size=(dim, b))

    # initializing optimization problem
    Model = op.Model_Test(dim, A, y, Anchors, delta_1, delta_2)

    # defining backward operators
    Proxs = st.create_Proxs(Anchors)

    # cases
    cases = 200

    # initialization
    w_init = np.zeros((b, dim))

    # defining N and K (randomly)
    Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
    F = np.sort(np.random.randint(1, f + 1, b - 2))
    F = np.hstack(([0], F, [f]))
    N, K = st.create_N_and_K(F, f, b, [0, 1], [0, 1])

    # defining Lap (randomly)
    G = nx.connected_watts_strogatz_graph(b, b - 2, p=.8, seed=0)
    Lap = nx.laplacian_matrix(G)
    Lap = Lap.toarray()
    norm = np.linalg.norm(Lap, 2)
    Lap = Lap / norm
    print(f'Algebraic Connectivity: {np.linalg.eigh(Lap)[0][1]}')

    # storage
    Vars = np.zeros((maxit, cases))
    Objs = np.zeros((maxit, cases))
    Spec = np.zeros(cases)

    for cs in tqdm(range(cases)):

        if cs == 0:
            new_norm = 0
            sLap = new_norm * Lap
        else:
            sLap = np.random.rand(b, b - 1)
            sLap = sLap - np.mean(sLap, axis=0)[np.newaxis, :]
            sLap = sLap @ sLap.T
            old_norm = np.linalg.norm(sLap, 2)
            new_norm = np.random.rand()
            sLap = new_norm * sLap / old_norm

        # running the two methods
        Vars[:, cs], Objs[:, cs] = \
            optim.General_Instance(Proxs, Grads, betas, F, w_init,
                                   maxit, tau, Model, Lap, sLap, N, K)

        Spec[cs] = new_norm

    show.plot_experiment_P(Objs, Spec, cases, maxit)
    return Objs, Spec, cases, maxit


def experiment_testing_betas(maxit=1000):
    '''
    NOTE Former experiment 0
    In this experiment, we test hetereogenity of data.
    '''

    np.random.seed(0)

    # problem and algorithm's parameters
    dim = 2       # dimension of the problem
    m = 20        # dimension of matrix (note: f must be <= than m)
    b = 4         # number of backward steps
    f = 15        # number of forward terms
    delta_1 = 1   # parameter for huber function
    delta_2 = 2
    tau = 1       # step-size

    # generating sample
    block_corruped_size = 2
    # np.random.seed(4)
    A = 2 * (np.random.rand(m, dim) - 0.5)
    noise_columns = np.random.randint(block_corruped_size, m)
    A[noise_columns, :] = 5 * A[noise_columns, :]
    y = np.random.rand(m)

    # generating anchor points
    Anchors = np.random.normal(0, 5, size=(dim, b))

    # initializing optimization problem
    Model = op.Model_Test(dim, A, y, Anchors, delta_1, delta_2)

    # defining backward operators
    Proxs = st.create_Proxs(Anchors)

    # cases
    cases = 20

    # initialization
    w_init = np.zeros((b, dim))

    # storage
    Vars_hom = np.zeros((maxit, cases))
    Objs_hom = np.zeros((maxit, cases))

    Vars_het = np.zeros((maxit, cases))
    Objs_het = np.zeros((maxit, cases))

    for cs in tqdm(range(cases)):

        # defining forward terms
        Grads, betas_het = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        betas_hom = np.max(betas_het) * np.ones(f)

        F = np.sort(np.random.randint(1, f + 1, b - 2))
        F = np.hstack(([0], F, [f]))

        # defining N and K with positive weights
        N, K = st.create_N_and_K(F, f, b, [0, 1], [0, 1])

        # running the two methods
        Vars_het[:, cs], Objs_het[:, cs] = \
            optim.Random_Instance(Proxs, Grads, betas_het, F, w_init,
                                  maxit, tau, Model, None, None, N, K)

        Vars_hom[:, cs], Objs_hom[:, cs] = \
            optim.Random_Instance(Proxs, Grads, betas_hom, F, w_init,
                                  maxit, tau, Model, None, None, N, K)


    show.plot_experiment_testing_betas(Objs_hom, Objs_het, cases, maxit)


def experiment_testing_W(maxit=100):
    '''
    NOTE: Former experiment_4

    In this experiment, we test the influence on the number of the spectral
    norm of W, with N and K chosen randomly and optimized.
    '''

    np.random.seed(0)

    # problem and algorithm's parameters
    dim = 2       # dimension of the problem
    m = 50        # dimension of matrix (note: f must be <= than m)
    b = 15         # number of backward steps
    delta_1 = 1   # parameter for huber function
    delta_2 = 2
    tau = 1       # step-size

    # generating sample
    block_corruped_size = 5
    # np.random.seed(4)
    A = 2 * (np.random.rand(m, dim) - 0.5)
    noise_columns = np.random.randint(block_corruped_size, m)
    noise_columns = np.arange(0, block_corruped_size)
    A[noise_columns, :] = 10 * A[noise_columns, :]
    y = np.random.rand(m)

    # generating anchor points
    Anchors = np.random.normal(0, 5, size=(dim, b))

    # initializing optimization problem
    Model = op.Model_Test(dim, A, y, Anchors, delta_1, delta_2)

    # defining backward operators
    Proxs = st.create_Proxs(Anchors)

    # cases
    cases = 10

    # initialization
    w_init = np.zeros((b, dim))

    # storage
    Objs = np.zeros((maxit, m, cases))
    Spects = np.zeros((m, cases))

    # defining Laplacian
    Lap =  b * np.eye(b) - np.ones(b)
    sLap = 0 * Lap

    for f in tqdm(range(1, m + 1)):

        # defining forward terms
        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)

        # splitting forward operators
        for cs in range(cases):

            F = np.sort(np.random.randint(1, f + 1, b - 2))
            F = np.hstack(([0], F, [f]))

            # defining N and K (randomly)
            N, K = st.create_N_and_K(F, f, b, [0, 1], [0, 1])

            # computing norm of W
            P = 1 / 4 * (N - K.T) @ np.diag(betas) @ (N.T - K)
            Spects[f - 1, cs] = np.linalg.norm(P, 2)

            _, Objs[:, f - 1, cs] = optim.General_Instance(Proxs, Grads, betas, F,
                                                       w_init, maxit, tau,
                                                       Model, Lap,
                                                       sLap, N, K)

    show.plot_experiment_W(m, cases, Objs, Spects, maxit)


def experiment_comparison_toy_example(hetereogenity=10, maxit=500):
    '''
    NOTE: former experiment_2_optimized
    In this experiment, we test our distributed method againts other instances
    in the literature, with optimized H and K
    '''

    np.random.seed(0)

    # problem and algorithm's parameters
    dim = 2        # dimension of the problem
    m = 20         # dimension of matrix (note: f must be <= than m)
    b = 5          # number of backward steps
    delta_1 = 1    # parameter for huber function
    delta_2 = 2
    tau = 1        # step-size

    # generating sample
    block_corruped_size = 2
    A = 2 * (np.random.rand(m, dim) - 0.5)
    noise_columns = np.random.randint(block_corruped_size, m)
    A[noise_columns, :] = hetereogenity * A[noise_columns, :]
    y = np.random.rand(m)

    # generating anchor points
    Anchors = np.random.normal(0, 5, size=(dim, b))

    # initializing optimization problem
    Model = op.Model_Test(dim, A, y, Anchors, delta_1, delta_2)

    # defining backward operators
    Proxs = st.create_Proxs(Anchors)

    # cases
    cases = 20

    # initialization
    w_init = np.zeros((b, dim))

    # storage
    Vars_aGFB = np.zeros((maxit, cases))
    Objs_aGFB = np.zeros((maxit, cases))

    Vars_SFB_plus = np.zeros((maxit, cases))
    Objs_SFB_plus = np.zeros((maxit, cases))

    Vars_ACL24 = np.zeros((maxit, cases))
    Objs_ACL24 = np.zeros((maxit, cases))

    Vars_AMTT23 = np.zeros((maxit, cases))
    Objs_AMTT23 = np.zeros((maxit, cases))

    Vars_BCLN23 = np.zeros((maxit, cases))
    Objs_BCLN23 = np.zeros((maxit, cases))

    for cs in tqdm(range(cases)):

        # Adapted Graph Forward-Backward (aGFB). Our paper
        if cs == 0:
            f = int(b * (b - 1) / 2)
        else:
            f = np.random.randint(1, b * (b - 1) / 2)

        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        Vars_aGFB[:, cs], Objs_aGFB[:, cs] = \
            optim.aGFB(Proxs, Grads, betas, w_init, maxit, tau, Model)

        # Split-Forward-Backward+ (SFB+). Our paper
        f = b - 1
        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        Vars_SFB_plus[:, cs], Objs_SFB_plus[:, cs] = \
            optim.SFB_plus(Proxs, Grads, betas, w_init, maxit, tau, Model)

        # Artacho, Campoy, Lopez-Pastor, 2024
        f = np.random.randint(1, b - 1 + 1)
        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        betas = np.max(betas) * np.ones(f)
        Vars_ACL24[:, cs], Objs_ACL24[:, cs] = \
            optim.ACL24(Proxs, Grads, betas, w_init, maxit, tau, Model)

        # Artacho, Malitsky, Tam, Torregrosa-Belén, 2023
        f = np.random.randint(1, b - 1 + 1)
        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        betas = np.max(betas) * np.ones(f)
        Vars_AMTT23[:, cs], Objs_AMTT23[:, cs] = \
            optim.AMTT23(Proxs, Grads, betas, w_init, maxit, tau, Model)

        # Bredies, Chenchene, Lorenz, Naldi, 2023
        f = np.random.randint(1, b - 1 + 1)
        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        betas = np.max(betas) * np.ones(f)
        Vars_BCLN23[:, cs], Objs_BCLN23[:, cs] = \
            optim.BCLN23(Proxs, Grads, betas, w_init, maxit, tau, Model)

    show.plot_experiment_comparison_toy_example(Objs_aGFB, Objs_SFB_plus,
                                                Objs_ACL24, Objs_AMTT23, Objs_BCLN23,
                                                hetereogenity, maxit)


def experiment_portfolio_optimization(maxit=500):
    '''
    In this experiment, we test our distributed method againts other instances
    in the literature on the portfolio optimization problem.
    '''

    tau = 1
    np.random.seed(0)

    # reading data and obtaining operators.
    # NOTE: Turn Download=True the first time running this code. Ensure to
    # have an API for key StockData.org.
    Proxs, Grads, betas, Model = port_opt.get_operators(4, Download=False)

    b = len(Proxs)

    # cases
    cases = 20

    # initialization
    w_init = np.zeros((b, Model.dim))

    # computing optimal solution
    _, _, _, x_opt = optim.SFB_plus(Proxs, Grads, betas, w_init, 20 * maxit,
                                    tau, Model, Compute_Dist_to_Sol=True)
    Model.x_opt = x_opt

    # storage
    Dist_aGFB = np.zeros((maxit, cases))
    Dist_SFB_plus = np.zeros((maxit, cases))
    Dist_ACL24 = np.zeros((maxit, cases))
    Dist_AMTT23 = np.zeros((maxit, cases))
    Dist_BCLN23 = np.zeros((maxit, cases))

    for cs in tqdm(range(cases)):

        # Adapted Graph Forward Backward (aGFB). Our paper
        _, _, Dist_aGFB[:, cs], _ = \
            optim.aGFB(Proxs, Grads, betas, w_init, maxit, tau, Model,
                      Compute_Dist_to_Sol=True)

        # Split-Forward-Backward+ (SFB+). Our paper
        _, _, Dist_SFB_plus[:, cs], _ = \
            optim.SFB_plus(Proxs, Grads, betas, w_init, maxit, tau, Model,
                                Compute_Dist_to_Sol=True)

        # Artacho, Campoy, Lopez-Pastor, 2024
        betas = np.max(betas) * np.ones(4)
        _, _, Dist_ACL24[:, cs], _ = \
            optim.ACL24(Proxs, Grads, betas, w_init, maxit, tau, Model,
                        Compute_Dist_to_Sol=True)

        # Artacho, Malitsky, Tam, Torregrosa-Belén, 2023
        betas = np.max(betas) * np.ones(4)
        _, _, Dist_AMTT23[:, cs], _ = \
            optim.AMTT23(Proxs, Grads, betas, w_init, maxit, tau, Model,
                         Compute_Dist_to_Sol=True)

        # Bredies, Chenchene, Lorenz, Naldi, 2023
        betas = np.max(betas) * np.ones(4)
        _, _, Dist_BCLN23[:, cs], _ = \
            optim.BCLN23(Proxs, Grads, betas, w_init, maxit, tau, Model,
                         Compute_Dist_to_Sol=True)

    show.plot_experiment_portopt(Dist_aGFB, Dist_SFB_plus, Dist_ACL24, Dist_AMTT23, Dist_BCLN23,
                                  x_opt, maxit)
