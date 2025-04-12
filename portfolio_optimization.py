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
This file contains useful functions to run the portfolio optimization
experiment in:

A. Akerman, E. Chenchene, P. Giselsson, E. Naldi.
Splitting the Forward-Backward Algorithm: A Full Characterization.
2025. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.

"""

import requests
import pandas as pd
import numpy as np
import structures as st
import operators as op
import plots as show
import pickle

def fetch_historical_data(symbol, start_date, end_date, api_key):
    '''
    Function to fetch historical data
    '''

    url = f'https://api.stockdata.org/v1/data/eod?symbols={symbol}&date_from' +\
        f'={start_date}&date_to={end_date}'

    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(url, headers=headers)
    data = response.json()

    if len(data['data']) != 0:
        df = pd.DataFrame(data['data'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df['close']

    else:
        print(f"Error fetching data for {symbol}")
        return pd.Series(dtype=float)


def get_data():

    # StockData.org API key (use your own)
    api_key = 'your API key'

    # define the date range
    start_date = '2020-01-01'
    end_date = '2021-01-01'

    # assets
    tickers = ["GOOGL", "AMZN", "AAPL", "BP", "MSFT", "NFLX"]

    # initialize a DataFrame to hold adjusted close prices
    pri = pd.DataFrame()

    # fetch data for each ticker
    for ticker in tickers:
        print(f'Retrieving data for: {ticker}')
        pri[ticker] = fetch_historical_data(ticker, start_date, end_date, api_key)

    # drop rows with missing values
    pri.dropna(inplace=True)

    # extract dates for plot
    dates = pri.index.to_list()
    dates = np.array(list(map(lambda x: f'0{x.month}.{x.year}', dates)))

    # turning into numpy
    pri = pri.to_numpy()

    # get revenue data
    rev = np.diff(pri, axis=0)
    rev = rev / pri[:-1, :] * 100

    # carbon intensity data (Le Guenedal and Roncalli 2022)
    CB_data = np.array([[0.460, 31.614, 44.275],    # Alphabet
                        [20.533, 19.606, 71.491],   # Amazon
                        [0.194, 3.314, 106.156],    # Apple
                        [177.714, 18.783, 375.077], # BP
                        [0.901, 28.262, 47.500],    # Microsoft
                        [1.909, 7.216, 94.277]      # Netflix
                        ])

    # saving data
    with open('data/dataset.npy', 'wb') as file:
        np.save(file, rev)
        np.save(file, CB_data)
        np.save(file, dates)

    return rev, CB_data, dates


def proj_halfspace(pars, level, x):
    '''
    Projecting onto the half space np.sum(pars * x) <= level.
    '''

    if np.sum(pars * x) <= level:
        return x
    else:
        return x - (np.sum(pars * x) - level) / np.sum(pars ** 2) * pars


def define_proj_halfspace(pars, level):
    '''
    Pythonic non-sense
    '''

    return lambda tau, x: proj_halfspace(pars, level, x)


def affine_map(A, b):

    return lambda x: A @ x - b


def projection_simplex(tau, vv, z=1):
    '''
    Projection onto standard simplex.
    '''
    v = np.copy(vv)
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)

    return w


def build_operators(rev, CB_data, f):
    '''
    Building proximity operators and forward steps for the portfolio
    optimization example.
    '''

    Proxs = []
    Grads = []
    betas = []

    T, n = rev.shape

    # initial position
    np.random.seed(0)
    init_position = np.random.rand(n)
    init_position = init_position / np.sum(init_position)

    # computing initial carbon presence
    init_carbon_lev = CB_data.T @ init_position

    # defining transition \ell_1 penalty
    Proxs.append(st.create_prox_function_ell_1(init_position))

    # simplex constraint
    Proxs.append(projection_simplex)

    Reduce = [0.01, 0.01, 0.08]
    for i in range(3):
        new_carbon_lev_i = (1 - Reduce[i]) * init_carbon_lev[i]
        Proxs.append(define_proj_halfspace(CB_data[:, i], new_carbon_lev_i))

    # defining forward terms splitting the volatility risk
    mean_rev = np.mean(rev, axis=0)
    nrev = rev - mean_rev[np.newaxis, :]
    NRevs = np.array_split(nrev, f, axis=0)

    for nrev_loc in NRevs:
        Cov_loc = nrev_loc.T @ nrev_loc / T

        Grads.append(affine_map(Cov_loc, mean_rev / f))
        betas.append(np.linalg.norm(Cov_loc, 2))

    return Proxs, Grads, betas


def get_operators(f, Download=False):

    if Download:
        rev, CB_data, dates = get_data()

    else:
        with open('data/dataset.npy', 'rb') as file:
            rev = np.load(file)
            CB_data = np.load(file)
            dates = np.load(file)

    # showing revenues
    show.plot_returns(rev, dates)

    T, dim = rev.shape

    mean_rev = np.mean(rev, axis=0)
    Cov = (rev - mean_rev[np.newaxis, :]).T @ (rev - mean_rev[np.newaxis, :])\
        / T

    # initializing portfolio model
    Model = op.Model_Portfolio(Cov, mean_rev, CB_data, dim)

    # getting proxs and grads
    Proxs, Grads, betas = build_operators(rev, CB_data, f)

    return Proxs, Grads, betas, Model
