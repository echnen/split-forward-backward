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
This file contains functions that are useful to generate the images in:

A. Akerman, E. Chenchene, P. Giselsson, E. Naldi.
Splitting the Forward-Backward Algorithm: A Full Characterization.
2025. DOI: 10.48550/arXiv.2504.10999.

For any comment, please contact: enis.chenchene@gmail.com
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


def plot_experiment_Lap(Objs, Spec, cases, maxit):

    # plotting objectives
    plt.figure(figsize=(5, 5))

    min_obj = np.min(Objs)
    plt.semilogy(Spec[0], Objs[-2, 0].T - min_obj, '.', color='r')
    plt.semilogy(Spec[1:20], Objs[-2, 1:20].T - min_obj, '.', color='b')
    plt.semilogy(Spec[21:], Objs[-2, 21:].T - min_obj, '.', color='k')
    plt.grid()
    plt.xlabel(r'Algebraic connectivity of $\mathcal{L}$')
    plt.ylabel(r'Final objective residual')
    plt.savefig('results/experiment_lap_obj.pdf', bbox_inches='tight')
    plt.show()


def plot_experiment_P(Objs, Spec, cases, maxit):

    # plotting objectives
    plt.figure(figsize=(5, 5))

    min_obj = np.min(Objs)
    plt.semilogy(Spec, Objs[-2, :].T - min_obj, '.', color='k')
    plt.grid()
    plt.xlabel(r'Spectral norm of $PP^T$')
    plt.ylabel(r'Final objective residual')
    plt.savefig('results/experiment_P_obj.pdf', bbox_inches='tight')
    plt.show()


def plot_experiment_testing_betas(Objs_hom, Objs_het, cases, maxit):

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
    plt.savefig('results/experiment_testing_betas.pdf', bbox_inches='tight')
    plt.show()


def plot_experiment_W(m, cases, Objs, Spects, maxit):

    # getting minimizers
    index = np.argmin(Spects)  # Flattened index
    (row_min, col_min) = np.unravel_index(index, Spects.shape)

    # custom colormap
    cmap = LinearSegmentedColormap.from_list("WhiteBlue", [(0, 0, 0),
                                                           (0, 0, 1),
                                                           (0.5, 0.9, 1)])
    norm = Normalize(np.min(Spects), np.max(Spects))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # plotting objectives
    fig = plt.figure(figsize=(5, 5))

    for f in range(m):
        for cs in range(cases):
            color = cmap(norm(Spects[f, cs]))
            plt.semilogy(Objs[:-10, f, cs] - np.min(Objs), color=color,
                         alpha=0.1)

    # plotting performance relative to optimal heuristic
    plt.semilogy(Objs[:-10, row_min, col_min] - np.min(Objs), color='r',
                 linewidth=3)

    plt.ylabel(r'$f(x^k) - \min \ f$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.xlim(0, maxit - 10)
    plt.grid()
    cbar_ax = fig.add_axes([0.38, 0.83, .5, 0.02])
    plt.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=r'$\|W\|_2$')
    plt.grid()
    plt.savefig('results/experiment_m_obj.pdf', bbox_inches='tight')
    plt.show()

    # plotting m vs spectral norm of W
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(1, m + 1), Spects, '.', color='k')
    plt.scatter(row_min + 1, Spects[row_min, col_min], color='r', s=20,
                zorder=2)
    plt.xlim(0, m)
    plt.xlabel(r'$m$')
    plt.ylabel(r'$\|W\|_2$')
    plt.grid()
    plt.savefig('results/experiment_m_sca.pdf', bbox_inches='tight')
    plt.show()


def plot_experiment_comparison_toy_example(Objs_aGFB, Objs_SFB_plus,
                                           Objs_ACL24, Objs_AMTT23,
                                           Objs_BCLN23, hetereogenity, maxit):

    maxit, cases = Objs_aGFB.shape

    # plotting objectives
    plt.figure(figsize=(5, 5))

    min_obj = min(np.min(Objs_aGFB), np.min(Objs_SFB_plus), np.min(Objs_ACL24),
                  np.min(Objs_AMTT23), np.min(Objs_BCLN23))

    # plotting all data
    plt.semilogy(Objs_aGFB - min_obj, color='k', alpha=.1)
    plt.semilogy(Objs_SFB_plus - min_obj, color='y', alpha=.1)

    plt.semilogy(Objs_ACL24 - min_obj, color='r', alpha=.1)
    plt.semilogy(Objs_AMTT23 - min_obj, color='b', alpha=.1)
    plt.semilogy(Objs_BCLN23 - min_obj, color='g', alpha=.1)

    # plotting averages
    if hetereogenity == 1:
        plt.semilogy(gmean(Objs_aGFB - min_obj, axis=1),
                     linewidth=3, color='k', alpha=1)
        plt.semilogy(gmean(Objs_SFB_plus - min_obj, axis=1),
                     linewidth=3, color='y', alpha=1)

        plt.semilogy(gmean(Objs_ACL24 - min_obj, axis=1),
                     linewidth=3, color='r', alpha=1)
        plt.semilogy(gmean(Objs_AMTT23 - min_obj, axis=1),
                     linewidth=3,  color='b', alpha=1)
        plt.semilogy(gmean(Objs_BCLN23 - min_obj, axis=1),
                     linewidth=3, color='g', alpha=1)

    else:
        plt.semilogy(gmean(Objs_aGFB - min_obj, axis=1),
                     linewidth=3, color='k', alpha=1, label='aGFB')
        plt.semilogy(gmean(Objs_SFB_plus - min_obj, axis=1),
                     linewidth=3, color='y', alpha=1, label='SFB+')
        plt.semilogy(gmean(Objs_ACL24 - min_obj, axis=1),
                     linewidth=3, color='r', alpha=1, label='GFB')
        plt.semilogy(gmean(Objs_AMTT23 - min_obj, axis=1),
                     linewidth=3, color='b', alpha=1, label='RFB')
        plt.semilogy(gmean(Objs_BCLN23 - min_obj, axis=1),
                     linewidth=3, color='g', alpha=1, label='SDY')

    plt.ylabel(r'$f(x^k) - \min \ f$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.xlim(0, maxit - 50)
    plt.grid()

    if hetereogenity != 1:
        plt.legend()

    if hetereogenity == 1:
        plt.savefig('results/experiment_comparison_toy_hom.pdf',
                    bbox_inches='tight')
    else:
        plt.savefig('results/experiment_comparison_toy_het.pdf',
                    bbox_inches='tight')

    plt.show()


def plot_experiment_portopt(Dist_aGFB, Dist_SFB_plus, Dist_ACL24, Dist_AMTT23,
                            Dist_BCLN23, x_opt, maxit):

    maxit, cases = Dist_aGFB.shape

    # plotting distances to solution
    plt.figure(figsize=(5, 5))

    # plotting all data
    # plt.semilogy(Dist_DFB_one, color='c', alpha=.1)
    plt.semilogy(Dist_aGFB, color='k', alpha=.1)
    plt.semilogy(Dist_SFB_plus, color='y', alpha=.1)
    plt.semilogy(Dist_ACL24, color='r', alpha=.1)
    plt.semilogy(Dist_AMTT23, color='b', alpha=.1)
    plt.semilogy(Dist_BCLN23, color='g', alpha=.1)

    # plotting averages
    plt.semilogy(gmean(Dist_aGFB, axis=1),
                 linewidth=3, color='k', alpha=1, label='aGFB')
    plt.semilogy(gmean(Dist_SFB_plus, axis=1),
                 linewidth=3, color='y', alpha=1, label='SFB+')
    plt.semilogy(gmean(Dist_ACL24, axis=1),
                 linewidth=3, color='r', alpha=1, label='GFB')
    plt.semilogy(gmean(Dist_AMTT23, axis=1),
                 linewidth=3, color='b', alpha=1, label='RFB')
    plt.semilogy(gmean(Dist_BCLN23, axis=1),
                 linewidth=3, color='g', alpha=1, label='SDY')

    plt.ylabel(r'$\| x_2^k - x^* \|^2$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.xlim(0, maxit - 50)
    plt.ylim(1e-21, 1e1)
    plt.grid()
    plt.legend()
    plt.savefig('results/experiment_portopt_dist.pdf', bbox_inches='tight')
    plt.show()


def plot_returns(rev, dates):

    T, dim = rev.shape
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # plt.figure(figsize=(7, 4))
    for i in range(2):
        axs[i].plot(np.flip(rev[:, i], axis=0), '-', linewidth=2, color='k',
                    alpha=1)
        axs[i].axvline(x=0, color='b')
        axs[i].axvline(x=31, color='b')
        axs[i].axvline(x=62, color='b')
        axs[i].axvline(x=93, color='b',)
        axs[i].axvline(x=T, color='b')
        axs[i].set_ylabel(r'Asset Return (\%)')
        axs[i].grid()

    domain = np.arange(0, 120, 19)
    axs[1].set_xticks(np.flip(domain), list(dates[domain]), rotation=45)
    axs[1].set_xlabel('Dates')
    axs[1].set_xlim(-0.05, T)

    plt.subplots_adjust(hspace=0)

    plt.savefig('results/dataset.pdf', bbox_inches='tight')
    plt.show()
