# -*- coding: utf-8 -*-

#############################################################
# 1. Imports
#############################################################

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math

from os import path

from scipy import stats
from astropy import units as u
from astropy import constants as const
from matplotlib import mlab
from matplotlib import patches
import decimal as dec

from scipy.optimize import curve_fit

# Allows LaTeX output in Jupyter and in matplotlib
sp.init_printing(use_latex=True, use_unicode=True)

#############################################################
# 2. Constants
#############################################################

PI = np.pi
TWO_PI = 2 * PI

kB = const.k_B
R = const.R
NA = const.N_A
G = const.G
g = const.g0
mu = 1.317e25 * G
R_E = 6371 * 1000
h = const.h
hbar = const.hbar
c = const.c
m_e = const.m_e
m_n = const.m_n
m_p = const.m_p
R_H = 2.18e-18 * u.J
q_e = const.e

ZERO = 0.00001

#############################################################
# 3. General Functions
#############################################################


def lsq(x, y):
    assert len(x) == len(y), 'Array dimensions do not match'
    n = float(len(x))  # Don't lose precision with int * float multiplication

    # compute covariance matrix and correlation coefficient of data
    cov = np.cov(x, y)
    varx = cov[0][0]
    vary = cov[1][1]
    sxy = cov[0][1]

    try:
        # Should help catch FPEs
        if varx == 0.0:
            varx = ZERO
        elif vary == 0.0:
            vary = ZERO

        if (np.sqrt(vary) is not np.sqrt(varx)):
            r = float(sxy) / (np.sqrt(vary) * np.sqrt(varx))
        else:
            r = np.sign(sxy) * 1
    except Exception as err:
        print(str(err))
        r = np.sign(sxy) * 1

    # lambda expression for a line
    # dummy parameter array of [1, 1]
    f = lambda x, *p: p[0] * x + p[1]
    pars = [1, 1]

    pvals, pcov = curve_fit(f, x, y, p0=pars)

    m, b = pvals
    sm = np.sqrt(pcov[0, 0])
    sb = np.sqrt(pcov[1, 1])
    sy = np.sqrt(vary)

    # y = mx + b; r is correlation
    return m, b, sy, sm, sb, r

#############################################################
# 4. Data
#############################################################

# Loading in all runs then concatenating all data to one array set
# HOWEVER, use these to find local maxes
# Note the usage of os.path.abspath(); this resolves any relative path
# errors and keeps track of potential bugs easier. Also, note the presence
# of 'franckhertz/'. Python tends to hoist all its loaded modules into the
# same executing directory as the parent script, so all file references
# must be made with respect to DataMaster's location.

volta_1, voltb_1 = np.loadtxt(path.abspath(
    './franckhertz/30vramp-1.txt'), skiprows=2, unpack=True)
volta_2, voltb_2 = np.loadtxt(path.abspath(
    './franckhertz/30vramp-2.txt'), skiprows=2, unpack=True)
volta_3, voltb_3 = np.loadtxt(path.abspath(
    './franckhertz/30vramp-3.txt'), skiprows=2, unpack=True)
volta_4, voltb_4 = np.loadtxt(path.abspath(
    './franckhertz/30vramp-4.txt'), skiprows=2, unpack=True)
volta_5, voltb_5 = np.loadtxt(path.abspath(
    './franckhertz/30vramp-5.txt'), skiprows=2, unpack=True)

# Do NOT use these to find local maxes
volta = np.concatenate((volta_1, volta_2, volta_3, volta_4, volta_5), axis=0)
voltb = np.concatenate((voltb_1, voltb_2, voltb_3, voltb_4, voltb_5), axis=0)

#############################################################
# 5. Lab-specific functions
#############################################################


def find_peaks(x, y):
    '''
    From two arrays [x] and [y], find the points of local maxima

    Returns an array of coords [ [x_1, y_1], [x_2, y_2], ..., [x_n, y_n] ] which are local maxima

    If no maxima found, return empty set []
    '''

    assert len(x) == len(y), 'Array dimensions must match'

    maxima = []

    for i in range(len(x)):
        # Firstly, check if x[i] is an extreme value of the set
        if i == 0:
            continue # We know that the first index of our data is not a maximum
        elif i == (len(x) - 1):
            break # Do not consider endpoint a maximum, even though it is a boundary

        else:
            if y[i-1] < y[i] and y[i+1] < y[i]:
                maxima.append([x[i], y[i]])

    if len(maxima) == 0:
        return maxima
    else:
        return np.array(maxima)

def get_peaks1():
    '''
    Returns the set of peaks for the first run
    '''

    return find_peaks(volta_1, voltb_1)

def get_peaks2():
    '''
    Returns the set of peaks for the second run
    '''

    return find_peaks(volta_2, voltb_2)

def plot_test():
    '''
    Test plotting function; checks if concatenated data resembles what we measured
    '''

    plt.plot(volta, voltb, 'r.')
