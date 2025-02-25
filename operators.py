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
This file contains an implementation of the general forward-backward operator.
For details see:

A. Akerman, E. Chenchene, P. Giselsson, E. Naldi.
Characterization of Nonexpansive Forward-Backward-type Algorithms with
Minimal Memory Requirements,
2025. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.

"""

import numpy as np


class genFBO:

    def __init__(self, tau, Proxs, Grads, dim, betas, Lap, sLap, N, K, F):

        self.tau = tau
        self.Proxs = Proxs
        self.Grads = Grads
        self.dim = dim
        self.Lap = Lap
        self.F = F
        self.N = N
        self.K = K
        self.b = len(Proxs)
        self.f = len(Grads)

        # building stL
        P = 1 / 4 * (N - K.T) @ np.diag(betas) @ (N.T - K)

        self.stL = -2 * (np.tril(Lap, k=-1) + np.tril(sLap, k=-1)
                         + np.tril(P, k=-1))
        self.diag = np.diag(Lap + sLap + P)

    def apply(self, w):

        x = np.zeros((self.b, self.dim))

        for i in range(self.b):

            # evaluating forward steps
            forward_i = np.zeros((self.f, self.dim))
            in_forward_i = self.K[:self.F[i], :] @ x

            for j in range(self.F[i]):
                forward_i[j, :] = self.Grads[j](in_forward_i[j, :])

            forward_i = self.N[i, :] @ forward_i

            # evaluating backward steps
            in_prox_i = (self.stL[i, :] @ x + w[i, :]
                         - self.tau * forward_i) / self.diag[i]

            x[i, :] = self.Proxs[i](self.tau / self.diag[i], in_prox_i)

        return w - self.Lap @ x, x
