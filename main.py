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
Run this script to reproduce all numerical experiments in Section REF of:

A. Akerman, E. Chenchene, P. Giselsson, E. Naldi.
Characterization of Nonexpansive Forward-Backward-type Algorithms with
Minimal Memory Requirements,
2025. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.

"""

import experiments as exp
import pathlib

if __name__ == '__main__':

    pathlib.Path("results").mkdir(parents=True, exist_ok=True)

    print('Running Experiment 0:\nTesting hetereogenity of data.')
    exp.experiment_0(maxit=1000)

    print('Running Experiment 1:\nTesting whether having negative' +
          ' (reflected-type) step sizes enhance the performance of the method.')
    exp.experiment_1(maxit=1000)
    
    print('Running Experiment 2:\nTesting againts other instances in' +
          'the literature.')
    exp.experiment_2(hetereogenity=1, maxit=500)
    exp.experiment_2(hetereogenity=10, maxit=500)

    print('Running Experiment 2:\nTesting againts other instances in' +
          'the literature.')
    exp.experiment_2_optimized(hetereogenity=1, maxit=500,heuristic=False)
    exp.experiment_2_optimized(hetereogenity=10, maxit=500, heuristic=False)

    print('Running Experiment 3:\nTesting randomly generated FB methods' +
          'with different H and K and studying the influence of ||W||.')
    exp.experiment_3(maxit=1000)

    print('Running Experiment 4:\nTesting if the number of forward terms' +
          ' influences the optimization performances.')
    exp.experiment_4(maxit=100)

    print('Running Experiment 4 but optimized:\nTesting if the number of forward terms' +
          ' influences the optimization performances.')
    #exp.experiment_4(maxit=100)
    exp.experiment_4_optimized(maxit=100, heuristic = False)
