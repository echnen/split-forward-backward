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
This file contains functions that are useful to generate the images in:

A. Akerman, E. Chenchene, P. Giselsson, E. Naldi.
Characterization of Nonexpansive Forward-Backward-type Algorithms with
Minimal Memory Requirements,
2025. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.

"""

import numpy as np
from scipy.stats.mstats import gmean

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

fonts = 15
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],
              'size': fonts})
rc('text', usetex=True)


def plot_experiment_0(Vars_hom, Vars_het, Objs_hom, Objs_het, cases, maxit):

    # plotting variances
    plt.figure(figsize=(5, 5))

    # plotting all data
    plt.semilogy(Vars_hom, color='r', alpha=.1)
    plt.semilogy(Vars_het, color='k', alpha=.1)

    # plotting averages
    plt.semilogy(gmean(Vars_hom, axis=1)[gmean(Vars_hom, axis=1) > 0],
                 color='r', linewidth=3, alpha=1)
    plt.semilogy(gmean(Vars_het, axis=1)[gmean(Vars_het, axis=1) > 0],
                 color='k', linewidth=3, alpha=1)

    plt.ylabel('Variance')
    plt.xlabel(r'Iteration number $(k)$')
    plt.xlim(0, maxit - 100)
    plt.grid()
    plt.savefig('results/experiment_0_1.pdf', bbox_inches='tight')
    plt.show()

    # plotting objectives
    plt.figure(figsize=(5, 5))

    min_obj = min(np.min(Objs_het), np.min(Objs_hom))
    Objs_res_hom = Objs_hom - min_obj
    Objs_res_het = Objs_het - min_obj

    # plotting all data
    plt.semilogy(Objs_res_hom, color='r', alpha=.1)
    plt.semilogy(Objs_res_het, color='k', alpha=.1)

    # plotting averages
    plt.semilogy(gmean(Objs_res_hom, axis=1)[gmean(Objs_res_hom, axis=1) > 0],
                 color='r', linewidth=3, alpha=1, label='Hom.')
    plt.semilogy(gmean(Objs_res_het, axis=1)[gmean(Objs_res_het, axis=1) > 0],
                 color='k', linewidth=3, alpha=1, label='Het.')

    plt.ylabel(r'$f(x^k) - \min \ f$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.xlim(0, maxit - 100)
    plt.grid()
    plt.legend()
    plt.savefig('results/experiment_0_2.pdf', bbox_inches='tight')
    plt.show()


def plot_experiment_1(Vars_pp, Vars_pn, Vars_np, Vars_nn,
                      Objs_pp, Objs_pn, Objs_np, Objs_nn, cases, maxit):

    # plotting variances
    plt.figure(figsize=(5, 5))

    plt.semilogy(Vars_pp, color='r', alpha=.1)
    plt.semilogy(Vars_pn, color='k', alpha=.1)
    plt.semilogy(Vars_np, color='b', alpha=.1)
    plt.semilogy(Vars_nn, color='g', alpha=.1)

    # averages
    plt.semilogy(gmean(Vars_pp, axis=1)[gmean(Vars_pp, axis=1) > 0],
                 color='r', linewidth=3, alpha=1)
    plt.semilogy(gmean(Vars_pn, axis=1)[gmean(Vars_pn, axis=1) > 0],
                 color='k', linewidth=3, alpha=1)
    plt.semilogy(gmean(Vars_np, axis=1)[gmean(Vars_np, axis=1) > 0],
                 color='b', linewidth=3, alpha=1)
    plt.semilogy(gmean(Vars_nn, axis=1)[gmean(Vars_nn, axis=1) > 0],
                 color='g', linewidth=3, alpha=1)

    plt.ylabel('Variance')
    plt.xlabel(r'Iteration number $(k)$')
    plt.xlim(0, maxit - 100)
    plt.grid()

    plt.savefig('results/experiment_1_1.pdf', bbox_inches='tight')
    plt.show()

    # plotting variances
    plt.figure(figsize=(5, 5))

    # plotting objectives
    min_obj = min(np.min(Objs_pp), np.min(Objs_pn), np.min(Objs_np),
                  np.min(Objs_nn))

    # plotting all data
    plt.semilogy(Objs_pp - min_obj, color='r', alpha=.1)
    plt.semilogy(Objs_pn - min_obj, color='k', alpha=.1)
    plt.semilogy(Objs_np - min_obj, color='b', alpha=.1)
    plt.semilogy(Objs_nn - min_obj, color='g', alpha=.1)

    # plotting means
    plt.semilogy(gmean(Objs_pp - min_obj, axis=1)[gmean(Objs_pp - min_obj, axis=1) > 0],
                 linewidth=3, color='r', alpha=1, label='PP')
    plt.semilogy(gmean(Objs_pn - min_obj, axis=1)[gmean(Objs_pn - min_obj, axis=1) > 0],
                 linewidth=3, color='k', alpha=1, label='PN')
    plt.semilogy(gmean(Objs_np - min_obj, axis=1)[gmean(Objs_np - min_obj, axis=1) > 0],
                 linewidth=3, color='b', alpha=1, label='NP')
    plt.semilogy(gmean(Objs_nn - min_obj, axis=1)[gmean(Objs_nn - min_obj, axis=1) > 0],
                 linewidth=3, color='g', alpha=1, label='NN')

    plt.ylabel(r'$f(x^k) - \min \ f$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.xlim(0, maxit - 100)
    plt.grid()
    plt.legend()
    plt.savefig('results/experiment_1_2.pdf', bbox_inches='tight')
    plt.show()


def plot_experiment_2(Vars_DFB, Vars_ACL24, Vars_AMTT23, Vars_BCLN23,
                      Objs_DFB, Objs_ACL24, Objs_AMTT23, Objs_BCLN23,
                      hetereogenity, maxit):

    maxit, cases = Vars_DFB.shape

    # plotting variances
    plt.figure(figsize=(5, 5))

    # plotting all data
    plt.semilogy(Vars_DFB, color='k', alpha=.1)
    plt.semilogy(Vars_ACL24, color='r', alpha=.1)
    plt.semilogy(Vars_AMTT23, color='b', alpha=.1)
    plt.semilogy(Vars_BCLN23, color='g', alpha=.1)

    # plotting averages
    plt.semilogy(gmean(Vars_DFB, axis=1)[gmean(Vars_DFB, axis=1) > 0],
                 linewidth=3, color='k', alpha=1)
    plt.semilogy(gmean(Vars_ACL24, axis=1)[gmean(Vars_ACL24, axis=1) > 0],
                 linewidth=3, color='r', alpha=1)
    plt.semilogy(gmean(Vars_AMTT23, axis=1)[gmean(Vars_AMTT23, axis=1) > 0],
                 linewidth=3, color='b', alpha=1)
    plt.semilogy(gmean(Vars_BCLN23, axis=1)[gmean(Vars_BCLN23, axis=1) > 0],
                 linewidth=3, color='g', alpha=1)

    plt.ylabel('Variance')
    plt.xlabel(r'Iteration number $(k)$')
    plt.xlim(0, maxit - 50)
    plt.grid()
    if hetereogenity == 1:
        plt.savefig('results/experiment_2_1_1.pdf', bbox_inches='tight')
    else:
        plt.savefig('results/experiment_2_1_2.pdf', bbox_inches='tight')

    plt.show()


    # plotting objectives
    plt.figure(figsize=(5, 5))

    min_obj = min(np.min(Objs_DFB), np.min(Objs_ACL24),
                  np.min(Objs_AMTT23), np.min(Objs_BCLN23))

    # plotting all data
    plt.semilogy(Objs_DFB - min_obj, color='k', alpha=.1)
    plt.semilogy(Objs_ACL24 - min_obj, color='r', alpha=.1)
    plt.semilogy(Objs_AMTT23 - min_obj, color='b', alpha=.1)
    plt.semilogy(Objs_BCLN23 - min_obj, color='g', alpha=.1)

    # plotting averages
    if hetereogenity == 1:
        plt.semilogy(gmean(Objs_DFB - min_obj, axis=1),
                     linewidth=3, color='k', alpha=1)
        plt.semilogy(gmean(Objs_ACL24 - min_obj, axis=1),
                     linewidth=3, color='r', alpha=1)
        plt.semilogy(gmean(Objs_AMTT23 - min_obj, axis=1),
                     linewidth=3,  color='b', alpha=1)
        plt.semilogy(gmean(Objs_BCLN23 - min_obj, axis=1),
                     linewidth=3, color='g', alpha=1)

    else:
        plt.semilogy(gmean(Objs_DFB - min_obj, axis=1),
                     linewidth=3, color='k', alpha=1, label='DFB')
        plt.semilogy(gmean(Objs_ACL24 - min_obj, axis=1),
                     linewidth=3, color='r', alpha=1, label='ACL24')
        plt.semilogy(gmean(Objs_AMTT23 - min_obj, axis=1),
                     linewidth=3, color='b', alpha=1, label='AMTT23')
        plt.semilogy(gmean(Objs_BCLN23 - min_obj, axis=1),
                     linewidth=3, color='g', alpha=1, label='BCLN23')

    plt.ylabel(r'$f(x^k) - \min \ f$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.xlim(0, maxit - 50)
    plt.grid()

    if hetereogenity != 1:
        plt.legend()

    if hetereogenity == 1:
        plt.savefig('results/experiment_2_2_1.pdf', bbox_inches='tight')
    else:
        plt.savefig('results/experiment_2_2_2.pdf', bbox_inches='tight')

    plt.show()


def plot_experiment_3(Vars, Objs, Lams, cases, maxit):

    # plotting variances
    plt.figure(figsize=(5, 5))

    cmap = LinearSegmentedColormap.from_list("WhiteBlue", [(0, 0, 0),
                                                           (0, 0, 1),
                                                           (0.5, 0.9, 1)])
    norm = Normalize(np.min(Lams), np.max(Lams))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for cs in range(cases):

        color = cmap(norm(Lams[cs]))
        plt.semilogy(Vars[:, cs], color=color, alpha=0.5)

    plt.ylabel('Variance')
    plt.xlabel(r'Iteration number $(k)$')
    plt.xlim(0, maxit - 100)
    plt.grid()
    plt.savefig('results/experiment_3_1.pdf', bbox_inches='tight')
    plt.show()

    # plotting objectives
    fig = plt.figure(figsize=(5, 5))
    for cs in range(cases):

        color = cmap(norm(Lams[cs]))
        plt.semilogy(Objs[:, cs] - np.min(Objs), color=color, alpha=0.5)

    plt.ylabel(r'$f(x^k) - \min \ f$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.xlim(0, maxit - 100)
    plt.grid()

    cbar_ax = fig.add_axes([0.38, 0.83, .5, 0.02])
    plt.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=r'$\|W\|_2$')
    plt.savefig('results/experiment_3_2.pdf', bbox_inches='tight')
    plt.show()


def plot_experiment_4(m, cases, Vars, Objs, Spects, maxit):

    # plotting variances
    plt.subplots(figsize=(5, 5))

    cmap = LinearSegmentedColormap.from_list("WhiteBlue", [(0, 0, 0),
                                                           (0, 0, 1),
                                                           (0.5, 0.9, 1)])
    norm = Normalize(vmin=1, vmax=m + 1, clip=True)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for f in range(1, m + 1):
        color = cmap(norm(f))
        plt.semilogy(Vars[:, f - 1, :], color=color, alpha=0.3)

    # plotting outliers
    plt.semilogy(Vars[:, 5, 2], color='r', linewidth=3, alpha=1)
    plt.semilogy(Vars[:, 9, 2], color='m', linewidth=3, alpha=1)

    plt.ylabel('Variance')
    plt.xlabel(r'Iteration number $(k)$')
    plt.xlim(0, maxit)
    plt.grid()
    plt.savefig('results/experiment_4_1.pdf', bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(5, 5))

    # plotting objectives
    for f in range(1, m + 1):
        color = cmap(norm(f))
        plt.semilogy(Objs[:, f - 1, :] - np.min(Objs), color=color, alpha=0.3)

    # plotting outliers
    plt.semilogy(Objs[:, 5, 2] - np.min(Objs), color='r', linewidth=3, alpha=1)
    plt.semilogy(Objs[:, 9, 2] - np.min(Objs), color='m', linewidth=3, alpha=1)

    plt.ylabel(r'$f(x^k) - \min \ f$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.xlim(0, maxit)
    plt.grid()
    cbar_ax = fig.add_axes([0.38, 0.83, .5, 0.02])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=r'$m$')
    plt.grid()
    plt.savefig('results/experiment_4_2.pdf', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(5, 5))
    # plotting m vs \|W\|
    for f in range(m):
        for cs in range(cases):
            if f == 5 and cs == 2:
                plt.scatter(f, Spects[f, cs], s=70, color='k')
                plt.scatter(f, Spects[f, cs], s=50, color='r')

            elif f == 9 and cs == 2:
                plt.scatter(9, Spects[9, 2], s=70, color='k')
                plt.scatter(9, Spects[9, 2], s=50, color='m')


            else:
                plt.scatter(f, Spects[f, cs], s=70, color='k', alpha=.5)
                plt.scatter(f, Spects[f, cs], s=50, color=[0.5, 0.5, 1], alpha=1)

    plt.xlabel(r'$m$')
    plt.ylabel(r'$\|W\|$')
    plt.grid()
    plt.savefig('results/experiment_4_3.pdf', bbox_inches='tight')
    plt.show()
