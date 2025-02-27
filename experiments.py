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
This file contains the numerical experiments in:

A. Akerman, E. Chenchene, P. Giselsson, E. Naldi.
Characterization of Nonexpansive Forward-Backward-type Algorithms with
Minimal Memory Requirements,
2025. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.

"""

import numpy as np
import structures as st
import operators as op
import optimization as optim
import plots as show
from tqdm import tqdm


def experiment_0(maxit=1000):
    '''
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

    # storing into dictionary
    data = {}
    data['dim'] = dim; data['A'] = A; data['y'] = y; data['Anchors'] = Anchors
    data['delta_1'] = delta_1; data['delta_2'] = delta_2

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
                                  maxit, tau, data, None, None, N, K)

        Vars_hom[:, cs], Objs_hom[:, cs] = \
            optim.Random_Instance(Proxs, Grads, betas_hom, F, w_init,
                                  maxit, tau, data, None, None, N, K)


    show.plot_experiment_0(Vars_hom, Vars_het, Objs_hom, Objs_het, cases, maxit)


def experiment_1(maxit=1000):
    '''
    In this experiment, we test whether having negative (reflected-type) step
    sizes enhance the performance of the method.

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
    block_corruped_size = 3
    # np.random.seed(4)
    A = 2 * (np.random.rand(m, dim) - 0.5)
    noise_columns = np.random.randint(block_corruped_size, m)
    A[noise_columns, :] = 5 * A[noise_columns, :]
    y = np.random.rand(m)

    # generating anchor points
    Anchors = np.random.normal(0, 5, size=(dim, b))

    # storing into dictionary
    data = {}
    data['dim'] = dim; data['A'] = A; data['y'] = y; data['Anchors'] = Anchors
    data['delta_1'] = delta_1; data['delta_2'] = delta_2

    # defining backward operators
    Proxs = st.create_Proxs(Anchors)

    # cases
    cases = 20

    # initialization
    w_init = np.zeros((b, dim))

    # storage
    Vars_pp = np.zeros((maxit, cases))
    Objs_pp = np.zeros((maxit, cases))

    Vars_pn = np.zeros((maxit, cases))
    Objs_pn = np.zeros((maxit, cases))

    Vars_np = np.zeros((maxit, cases))
    Objs_np = np.zeros((maxit, cases))

    Vars_nn = np.zeros((maxit, cases))
    Objs_nn = np.zeros((maxit, cases))

    for cs in tqdm(range(cases)):

        # defining forward terms
        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)

        F = np.sort(np.random.randint(1, f + 1, b - 2))
        F = np.hstack(([0], F, [f]))

        # positive N and positive K
        Vars_pp[:, cs], Objs_pp[:, cs] = \
            optim.Random_Instance(Proxs, Grads, betas, F,
                                  w_init, maxit, tau, data, [0, 1], [0, 1])

        # positive N and negative K
        Vars_pn[:, cs], Objs_pn[:, cs] = \
            optim.Random_Instance(Proxs, Grads, betas, F,
                                  w_init, maxit, tau, data, [0, 1], [-1, 1])

        # negative N and positive K
        Vars_np[:, cs], Objs_np[:, cs] = \
            optim.Random_Instance(Proxs, Grads, betas, F,
                                  w_init, maxit, tau, data, [-1, 1], [0, 1],)

        # negative N and negative K
        Vars_nn[:, cs], Objs_nn[:, cs] = \
            optim.Random_Instance(Proxs, Grads, betas, F,
                                  w_init, maxit, tau, data, [-1, 1], [-1, 1])

    show.plot_experiment_1(Vars_pp, Vars_pn, Vars_np, Vars_nn, Objs_pp, Objs_pn,
                           Objs_np, Objs_nn, cases, maxit)


def experiment_2(hetereogenity=10, maxit=500):
    '''
    In this experiment, we test our distributed method againts other instances
    in the literature.
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

    # storing into dictionary
    data = {}
    data['dim'] = dim; data['A'] = A; data['y'] = y; data['Anchors'] = Anchors
    data['delta_1'] = delta_1; data['delta_2'] = delta_2

    # defining backward operators
    Proxs = st.create_Proxs(Anchors)

    # cases
    cases = 20

    # initialization
    w_init = np.zeros((b, dim))

    # storage
    Vars_DFB = np.zeros((maxit, cases))
    Objs_DFB = np.zeros((maxit, cases))

    Vars_ACL24 = np.zeros((maxit, cases))
    Objs_ACL24 = np.zeros((maxit, cases))

    Vars_AMTT23 = np.zeros((maxit, cases))
    Objs_AMTT23 = np.zeros((maxit, cases))

    Vars_BCLN23 = np.zeros((maxit, cases))
    Objs_BCLN23 = np.zeros((maxit, cases))

    for cs in tqdm(range(cases)):

        # Distributed Forward Backward (DFB). Our paper
        if cs == 0:
            f = int(b * (b - 1) / 2)
        else:
            f = np.random.randint(1, b * (b - 1) / 2)

        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        Vars_DFB[:, cs], Objs_DFB[:, cs] = \
            optim.DFB(Proxs, Grads, betas, w_init, maxit, tau, data)



        # Artacho, Campoy, Lopez-Pastor, 2024
        f = np.random.randint(1, b - 1 + 1)
        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        # betas = np.max(betas) * np.ones(f)
        Vars_ACL24[:, cs], Objs_ACL24[:, cs] = \
            optim.ACL24(Proxs, Grads, betas, w_init, maxit, tau, data)

        # Artacho, Malitsky, Tam, Torregrosa-Belén, 2023
        f = np.random.randint(1, b - 1 + 1)
        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        betas = np.max(betas) * np.ones(f)
        Vars_AMTT23[:, cs], Objs_AMTT23[:, cs] = \
            optim.AMTT23(Proxs, Grads, betas, w_init, maxit, tau, data)

        # Bredies, Chenchene, Lorenz, Naldi, 2023
        f = np.random.randint(1, b - 1 + 1)
        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        betas = np.max(betas) * np.ones(f)
        Vars_BCLN23[:, cs], Objs_BCLN23[:, cs] = \
            optim.BCLN23(Proxs, Grads, betas, w_init, maxit, tau, data)

    show.plot_experiment_2(Vars_DFB, Vars_ACL24, Vars_AMTT23, Vars_BCLN23,
                           Objs_DFB, Objs_ACL24, Objs_AMTT23, Objs_BCLN23,
                           hetereogenity, maxit)

def experiment_2_optimized(hetereogenity=10, maxit=500, heuristic = False):
    '''
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

    # storing into dictionary
    data = {}
    data['dim'] = dim; data['A'] = A; data['y'] = y; data['Anchors'] = Anchors
    data['delta_1'] = delta_1; data['delta_2'] = delta_2

    # defining backward operators
    Proxs = st.create_Proxs(Anchors)

    # cases
    cases = 20

    # initialization
    w_init = np.zeros((b, dim))

    # storage
    Vars_DFB = np.zeros((maxit, cases))
    Objs_DFB = np.zeros((maxit, cases))

    Vars_DFB_opt = np.zeros((maxit, cases))
    Objs_DFB_opt = np.zeros((maxit, cases))

    Vars_ACL24 = np.zeros((maxit, cases))
    Objs_ACL24 = np.zeros((maxit, cases))

    Vars_AMTT23 = np.zeros((maxit, cases))
    Objs_AMTT23 = np.zeros((maxit, cases))

    Vars_BCLN23 = np.zeros((maxit, cases))
    Objs_BCLN23 = np.zeros((maxit, cases))

    for cs in tqdm(range(cases)):

        # Distributed Forward Backward (DFB). Our paper
        if cs == 0:
            f = int(b * (b - 1) / 2)
        else:
            f = np.random.randint(1, b * (b - 1) / 2)

        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        Vars_DFB[:, cs], Objs_DFB[:, cs] = \
            optim.DFB(Proxs, Grads, betas, w_init, maxit, tau, data)

        # Distributed Forward Backward (DFB). Our paper
        if cs == 0:
            f = int(b * (b - 1) / 2)
        else:
            f = np.random.randint(1, b - 1 + 1)
        f = b-1
        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        Vars_DFB_opt[:, cs], Objs_DFB_opt[:, cs] = \
            optim.DFB_optimized(Proxs, Grads, betas, w_init, maxit, tau, data,heuristic=heuristic)



        # Artacho, Campoy, Lopez-Pastor, 2024
        f = np.random.randint(1, b - 1 + 1)
        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        # betas = np.max(betas) * np.ones(f)
        Vars_ACL24[:, cs], Objs_ACL24[:, cs] = \
            optim.ACL24(Proxs, Grads, betas, w_init, maxit, tau, data)

        # Artacho, Malitsky, Tam, Torregrosa-Belén, 2023
        f = np.random.randint(1, b - 1 + 1)
        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        betas = np.max(betas) * np.ones(f)
        Vars_AMTT23[:, cs], Objs_AMTT23[:, cs] = \
            optim.AMTT23(Proxs, Grads, betas, w_init, maxit, tau, data)

        # Bredies, Chenchene, Lorenz, Naldi, 2023
        f = np.random.randint(1, b - 1 + 1)
        Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)
        betas = np.max(betas) * np.ones(f)
        Vars_BCLN23[:, cs], Objs_BCLN23[:, cs] = \
            optim.BCLN23(Proxs, Grads, betas, w_init, maxit, tau, data)

    show.plot_experiment_2_optimized(Vars_DFB, Vars_DFB_opt, Vars_ACL24, Vars_AMTT23, Vars_BCLN23,
                           Objs_DFB,Objs_DFB_opt, Objs_ACL24, Objs_AMTT23, Objs_BCLN23,
                           hetereogenity, maxit)



def experiment_3(maxit=1000):
    '''
    In this experiment, we consider randomly generated FB methods with
    different N and K and study the influence of the spectrum of P
    '''

    np.random.seed(0)

    # problem and algorithm's parameters
    dim = 2        # dimension of the problem
    m = 20         # dimension of matrix (note: f must be <= than m)
    b = 4          # number of backward steps
    f = 20         # number of forward steps
    delta_1 = 1    # parameter for huber function
    delta_2 = 2
    tau = 1        # step-size

    # generating sample
    np.random.seed(4)
    A = np.random.rand(m, dim)
    A[0, :] = 6 * A[0, :]
    y = np.random.rand(m)

    # generating anchor points
    Anchors = np.random.normal(0, 5, size=(dim, b))

    # defining backward operators
    Proxs = st.create_Proxs(Anchors)

    # defining forward terms
    Grads, betas = st.create_Grads_hub_flat(delta_1, delta_2, f, A, y)

    # cases
    cases = 50

    # initialization
    w_init = np.zeros((b, dim))

    # storage
    Vars = np.zeros((maxit, cases))
    Objs = np.zeros((maxit, cases))
    Lams = np.zeros(cases)

    # defining Laplacian
    Lap =  b * np.eye(b) - np.ones(b)
    sLap = 0 * Lap

    for cs in tqdm(range(cases)):

        # splitting forward operators
        F = np.sort(np.random.randint(1, f + 1, b - 2))
        F = np.hstack(([0], F, [f]))

        # defining N and K
        N, K = st.create_N_and_K(F, f, b, [0, 1], [0, 1])

        # computing spectrum of forward matrix
        P = 1 / 4 * (N - K.T) @ np.diag(betas) @ (N.T - K)
        lambda_1 = np.linalg.norm(P, 2)
        Lams[cs] = lambda_1

        # defining operator
        J = op.genFBO(tau, Proxs, Grads, dim, betas, Lap, sLap, N, K, F)

        # optimization
        w = np.copy(w_init)

        for k in range(maxit):

            # step
            w, x = J.apply(w)
            mean_x = np.mean(x, axis=0)

            # compute objective value
            obj = st.fobj_exp1(mean_x, A, y, Anchors, delta_1, delta_2)
            Objs[k, cs] = obj

            # computing variance
            var = np.sum((x - mean_x[np.newaxis, :]) ** 2)
            Vars[k, cs] = var

    show.plot_experiment_3(Vars, Objs, Lams, cases, maxit)


def experiment_4(maxit=200):
    '''
    In this experiment, we test the influence on the number of forward terms
    actually. Does it help the optimization?
    '''

    np.random.seed(0)

    # problem and algorithm's parameters
    dim = 2       # dimension of the problem
    m = 20        # dimension of matrix (note: f must be <= than m)
    b = 4         # number of backward steps
    delta_1 = 1   # parameter for huber function
    delta_2 = 2
    tau = 1       # step-size

    # generating sample
    block_corruped_size = 1
    # np.random.seed(4)
    A = 2 * (np.random.rand(m, dim) - 0.5)
    noise_columns = np.random.randint(block_corruped_size, m)
    noise_columns = np.arange(0, block_corruped_size)
    A[noise_columns, :] = 4 * A[noise_columns, :]
    y = np.random.rand(m)

    # generating anchor points
    Anchors = np.random.normal(0, 5, size=(dim, b))

    # defining backward operators
    Proxs = st.create_Proxs(Anchors)

    # cases
    cases = 3

    # initialization
    w_init = np.zeros((b, dim))

    # storage
    Vars = np.zeros((maxit, m, cases))
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

            # defining N and K
            N, K = st.create_N_and_K(F, f, b, [0, 1], [0, 1])

            # computing norm of W
            P = 1 / 4 * (N - K.T) @ np.diag(betas) @ (N.T - K)
            Spects[f - 1, cs] = np.linalg.norm(P, 2)

            # defining operator
            J = op.genFBO(tau, Proxs, Grads, dim, betas, Lap, sLap, N, K, F)

            # optimization
            w = np.copy(w_init)

            for k in range(maxit):

                # step
                w, x = J.apply(w)
                mean_x = np.mean(x, axis=0)

                # compute objective value
                obj = st.fobj_exp1(mean_x, A, y, Anchors, delta_1, delta_2)
                Objs[k, f - 1, cs] = obj

                # computing variance
                var = np.sum((x - mean_x[np.newaxis, :]) ** 2)
                Vars[k, f - 1, cs] = var

    show.plot_experiment_4(m, cases, Vars, Objs, Spects, maxit)

def experiment_4_optimized(maxit=200,heuristic = False):
    '''
    In this experiment, we test the influence on the number of forward terms
    actually, when N and K are optimized. Does it help the optimization?
    '''

    np.random.seed(0)

    # problem and algorithm's parameters
    dim = 2       # dimension of the problem
    m = 20        # dimension of matrix (note: f must be <= than m)
    b = 4         # number of backward steps
    delta_1 = 1   # parameter for huber function
    delta_2 = 2
    tau = 1       # step-size

    # generating sample
    block_corruped_size = 1
    # np.random.seed(4)
    A = 2 * (np.random.rand(m, dim) - 0.5)
    noise_columns = np.random.randint(block_corruped_size, m)
    noise_columns = np.arange(0, block_corruped_size)
    A[noise_columns, :] = 4 * A[noise_columns, :]
    y = np.random.rand(m)

    # generating anchor points
    Anchors = np.random.normal(0, 5, size=(dim, b))

    # defining backward operators
    Proxs = st.create_Proxs(Anchors)

    # cases
    cases = 3

    # initialization
    w_init = np.zeros((b, dim))

    # storage
    Vars = np.zeros((maxit, m, cases))
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
            if heuristic:
                F = optim.F_heuristic(f,b,betas)
            else:
                F = [0] + [f//(b-1)*i for i in range(1,b-1)]+  [f]

            # defining N and K
            N, K = st.create_N_and_K_optimized(F, f, b, np.diag(betas))

            # computing norm of W
            P = 1 / 4 * (N - K.T) @ np.diag(betas) @ (N.T - K)
            Spects[f - 1, cs] = np.linalg.norm(P, 2)

            # defining operator
            J = op.genFBO(tau, Proxs, Grads, dim, betas, Lap, sLap, N, K, F)

            # optimization
            w = np.copy(w_init)

            for k in range(maxit):

                # step
                w, x = J.apply(w)
                mean_x = np.mean(x, axis=0)

                # compute objective value
                obj = st.fobj_exp1(mean_x, A, y, Anchors, delta_1, delta_2)
                Objs[k, f - 1, cs] = obj

                # computing variance
                var = np.sum((x - mean_x[np.newaxis, :]) ** 2)
                Vars[k, f - 1, cs] = var

    show.plot_experiment_4(m, cases, Vars, Objs, Spects, maxit, outliers = [1,3])
