#############################################################
#
# lab1 analysis
#
# 1. Imports
#############################################################
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from astropy import units as u
from astropy import constants as const
from sympy import *
import decimal as dec

from scipy.optimize import curve_fit
import psalib as pl

# Allows LaTeX output in Jupyter and in matplotlib
init_printing(use_latex=True, use_unicode=True)

#############################################################
# 2. Constants
#############################################################

PI     = np.pi
TWO_PI = 2*PI

kB   = const.k_B
R    = const.R
NA   = const.N_A
G    = const.G
g    = const.g0
µ    = 1.317e25 * G
R_E  = 6371 * 1000
h    = const.h
hbar = const.hbar
c    = const.c
m_e  = const.m_e
m_n  = const.m_n
m_p  = const.m_p
R_H  = 2.18e-18 * u.J

#############################################################
# 3. General Functions
#############################################################

def lsq(x, y):
    assert len(x) == len(y), 'Array dimensions do not match'
    n = float(len(x)) # Don't lose precision with int * float multiplication

    # compute covariance matrix and correlation coefficient of data
    cov  = np.cov(x, y)
    varx = cov[0][0]
    vary = cov[1][1]
    sxy  = cov[0][1]
    r    = sxy / (np.sqrt(vary) * np.sqrt(varx))

    # lambda expression for a line
    # dummy parameter array of [1, 1]
    f    = lambda x, *p: p[0]*x + p[1]
    pars = [1, 1]

    pvals, pcov = curve_fit(f, x, y, p0=pars)

    m, b = pvals
    sm   = np.sqrt(pcov[0, 0])
    sb   = np.sqrt(pcov[1, 1])
    sy   = np.sqrt(vary)

    # y = mx + b; r is correlation
    return m, b, sy, sm, sb, r

def gennoise(N):
    x = np.random.uniform(-1/2, 1/2, N)
    y = np.random.uniform(-1/2, 1/2, N)

    return x, y

def piEstimate(x, y):
    r = 0.5 # hardcode for simplicity

    s = 0

    for i in range(len(x)):
        if ((x[i]**2 + y[i]**2)**(1/2) <= r):
            s += 1

    return 4 * (float(s) / float(len(x)))

def build(N):

    ret = np.zeros(N)

    for i in range(N):
        _x, _y = gennoise(2**(i+1))

        pi     = piEstimate(_x, _y)
        delPi  = np.abs(pi - PI) / PI

        ret[i] = delPi

    return ret

#############################################################
# 4. Plotting Functions
#############################################################
def plot_volts_cps(a, b):
    plt.plot(a, b)
    plt.show()

def plot_times_counts(t, cs):
    x = np.linspace(0, 4, 1000)

    m, b, sy, sm, sb, r = pl.lsq(t, np.log(cs))

    plt.plot(t, np.log(cs), 'r.')
    plt.plot(x, m*x + b, 'b-')
    plt.show()

#############################################################
# 5. Program Main
#
#    Put all the program execution code below here
#############################################################

# I'll organize and nicely name these vars in a bit
volts = np.array([315, 335, 355, 375, 395, 415, 435, 455, 475])
cps   = np.array([11.8, 71.3, 73.2, 75.1, 76.4, 74.3, 75.0, 75.6, 79.7])
c     = np.array([49, 50, 55, 48, 45])

times  = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
counts = np.array([163, 138, 137, 114, 103, 88, 86, 80, 67])

# Ok now to draw some shit

# plot_volts_cps(volts, cps)
plot_times_counts(times, counts)

# @end
