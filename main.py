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
Run this script to reproduce all numerical experiments in Section REF of:

A. Akerman, E. Chenchene, P. Giselsson, E. Naldi.
Splitting the Forward-Backward Algorithm: A Full Characterization.
2025. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.

"""

import experiments as exp
import pathlib

if __name__ == '__main__':

    # pathlib.Path("results").mkdir(parents=True, exist_ok=True)

    print('Running Experiment 1:\nTesting the influence of M.')
    exp.experiment_testing_M(maxit=100)

    print('\n\nRunning Experiment 2:\nTesting the influence of P.')
    exp.experiment_testing_P(maxit=100)

    print('\n\nRunning Experiment 3:\nTesting hetereogenity of data.')
    exp.experiment_testing_betas(maxit=1000)

    print('\n\nRunning Experiment 4:\nTesting the influence of W.')
    exp.experiment_testing_W(maxit=100)

    print('\n\nRunning Experiment 5:\nTesting against other instances in' +
          'the literature: The case of a toy example.')
    exp.experiment_comparison_toy_example(hetereogenity=1, maxit=500)
    exp.experiment_comparison_toy_example(hetereogenity=10, maxit=500)

    print('\n\nRunning Experiment 6:\nTesting against other instances in' +
          'the literature: Portfolio optimization with decarbonization.')
    exp.experiment_portfolio_optimization(maxit=500)
