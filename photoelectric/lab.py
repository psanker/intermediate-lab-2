# -*- coding: utf-8 -*-

#############################################################
# 1. Imports
#############################################################

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math

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

PI     = np.pi
TWO_PI = 2*PI

kB   = const.k_B
R    = const.R
NA   = const.N_A
G    = const.G
g    = const.g0
mu    = 1.317e25 * G
R_E  = 6371 * 1000
h    = const.h
hbar = const.hbar
c    = const.c
m_e  = const.m_e
m_n  = const.m_n
m_p  = const.m_p
R_H  = 2.18e-18 * u.J
q_e  = const.e

ZERO = 0.00001

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
    f    = lambda x, *p: p[0]*x + p[1]
    pars = [1, 1]

    pvals, pcov = curve_fit(f, x, y, p0=pars)

    m, b = pvals
    sm   = np.sqrt(pcov[0, 0])
    sb   = np.sqrt(pcov[1, 1])
    sy   = np.sqrt(vary)

    # y = mx + b; r is correlation
    return m, b, sy, sm, sb, r

def exponential_limit_fit(x, y):
    assert len(x) == len(y), 'Array dimensions do not match'

    # compute covariance matrix and correlation coefficient of data
    cov  = np.cov(x, y)
    varx = cov[0][0]
    vary = cov[1][1]
    sxy  = cov[0][1]

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
    f    = lambda x, *p: p[0] - p[1]*np.exp(-1*p[2]*x)
    pars = [1, 1, 1]

    pvals, pcov = curve_fit(f, x, y, p0=pars)

    A, B, l = pvals

    sA = np.sqrt(pcov[0][0])
    sB = np.sqrt(pcov[1][1])
    sl = np.sqrt(pcov[2][2])

    return A, sA, B, sB, l, sl, r

#############################################################
# 4. Data
#############################################################

# 4358 Wavelength

wavelength_4358_V_1 = np.array([ZERO, .68, .78, .85, .95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85])
wavelength_4358_d_1 = np.array([-60, 3, 5, 6, 7, 8, 8, 9, 9, 9, 9, 9, 9, 9])

wavelength_4358_V_2 = np.array([ZERO, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
wavelength_4358_d_2 = np.array([-61, -5, ZERO, 3, 5, 7, 8, 8, 8, 9, 9, 9, 9, 9, 9])

  # Appended for complete data set
wavelength_4358_V = np.append(wavelength_4358_V_1, wavelength_4358_V_2)
wavelength_4358_d = np.append(wavelength_4358_d_1, wavelength_4358_d_2)

# 546 Wavelength

wavelength_546_V_1 = np.array([ZERO, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4])
wavelength_546_d_1 = np.array([-23, -10, -5, -1, ZERO, 1, 1.5, 2, 2, 2, 2, 2, 2, 2])

wavelength_546_V_2 = np.array([ZERO, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4])
wavelength_546_d_2 = np.array([-21, -10, -5, -1, ZERO, 1, 2, 2, 2, 2, 2, 2, 2, 2])

# Appended for complete data set
wavelength_546_V = np.append(wavelength_546_V_1, wavelength_546_V_2)
wavelength_546_d = np.append(wavelength_546_d_1, wavelength_546_d_2)

# 577 Wavelength -- The two runs are identical

wavelength_577_V_1 = np.array([ZERO, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
wavelength_577_d_1 = np.array([-5, -2, -1, -0.5, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO])

wavelength_577_V_2 = np.array([ZERO, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
wavelength_577_d_2 = np.array([-5, -2, -1, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO])
# Appended for complete data set
wavelength_577_V = np.append(wavelength_577_V_1, wavelength_577_V_2)
wavelength_577_d = np.append(wavelength_577_d_1, wavelength_577_d_2)

d_deflection     = 1 # in millimeters

#############################################################
# 5. Lab-specific functions
#############################################################

def find_limit_asymptote(x, y, tolerance=0.05):
    # for data set {x,y}, find a linear approximation of the tail
    # needs at least 3 data points to function safely

    assert len(x) == len(y), 'Array dimensions do not match'

    m, b, sy, sm, sb, r = (0., 0., 0., 0., 0., 0.)
    run = True

    set_x = x
    set_y = y

    while (run):
        if len(set_x) < 3:
            print('Could not find sufficient asymptote approximation; widen tolerance?')
            return np.zeros(6)

        # generate lsq line
        m, b, sy, sm, sb, r = lsq(set_x, set_y)

        # find weighted deviations
        diffs    = ((m*set_x + b) - set_y)**2 / set_y**2
        meandiff = np.sum(np.sqrt(diffs)) / float(len(diffs))

        if meandiff > tolerance:
            set_x = set_x[1:]
            set_y = set_y[1:]
        else:
            run = False

    return np.array([m, b, sy, sm, sb, r])

def find_stopping_potential(x, y, tolerance=[1, 0.5, 0.25], limtol=0.05):
    A, sA, B, sB, l, sl, r = exponential_limit_fit(x, y)
    lim                    = find_limit_asymptote(x, y, limtol)

    xmin, xmax = (x[0], x[-1])
    xset = np.linspace(xmin, xmax, 1000)

    voltage_guess = -1. * np.ones(len(tolerance))

    for i in range(len(tolerance)):

        for j in range(len(xset)):
            f1 = A - B*np.exp(-1*l*xset[j])
            f2 = lim[0]*xset[j] + lim[1]

            diff = np.sqrt((f2 - f1)**2.)

            # scale tolerance with respect to deviation of line
            if diff < tolerance[i]*lim[2]:
                voltage_guess[i] = xset[j]
                break

    return voltage_guess

def plot_4358():
    A, sA, B, sB, l, sl, r = exponential_limit_fit(wavelength_4358_V, wavelength_4358_d)
    lim                    = find_limit_asymptote(wavelength_4358_V, wavelength_4358_d, tolerance=0.05)

    sp    = find_stopping_potential(wavelength_4358_V, wavelength_4358_d)
    sp_mu = np.mean(sp)
    sp_s  = np.std(sp)

    x  = np.linspace(0, 1.9, 1000)

    plt.figure()
    plt.plot(x, A - B*np.exp(-1*l*x), 'b--', label=('Current'))

    if lim is not np.zeros(6):
        plt.plot(x, lim[0]*x + lim[1], 'g--', label='Limit')

    plt.errorbar(wavelength_4358_V, wavelength_4358_d, yerr=d_deflection, fmt='r.', ecolor='k', alpha=0.4)
    plt.errorbar(sp_mu, A - B*np.exp(-1*l*sp_mu), xerr=sp_s, yerr=lim[2], fmt='go', ecolor='k', label=('%1.3f $\pm$ %1.3e$V$' % (sp_mu, sp_s)))

    plt.xlabel('Voltage ($V$)')
    plt.ylabel('Deflection ($mm$)')
    plt.legend(loc='lower right')

    plt.annotate('$f(x)=A + Be^{-\\lambda x}$\n$A=$%f%f\n$B=$%f$\pm$%f\n$\\lambda=$%f$\pm$%f' % (A, sA, B, sB, l, sl), xy=(1, 5), xytext=(1, -20), arrowprops=dict(facecolor='black', headwidth=6, width=.2, shrink=0.05))
    plt.annotate('$y=mx + b$\n$m=$%f$\pm$%f\n$b=$%f$\pm$%f\n$r=$%f' % (lim[0], lim[3], lim[1], lim[4], lim[5]), xy=(0.2, 6), xytext=(0.5, -35), arrowprops=dict(facecolor='black', headwidth=6, width=.2, shrink=0.05))

# Unsure if this is useful at all
# def plot_4358_corrected():
#     A, sA, B, sB, l, sl, r = exponential_limit_fit(wavelength_4358_V, wavelength_4358_d)
#     lim                    = find_limit_asymptote(wavelength_4358_V, wavelength_4358_d, tolerance=0.05)
#
#     x = np.linspace(0, 1.9, 1000)
#
#     plt.figure()
#     plt.errorbar(wavelength_4358_V, wavelength_4358_d, yerr=d_deflection, fmt='r.', ecolor='k', alpha=0.4)
#
#     plt.plot(x, -1*(A+B*np.exp(-1*l*x)) - (-1*(lim[0]*x + lim[1])), 'b-', label='Corrected current')
#
#     plt.xlabel('Voltage ($V$)')
#     plt.ylabel('Deflection ($mm$)')
#     plt.legend(loc='lower right')

def plot_546():
    np.seterr(invalid='ignore') # Dirty, but whatever

    A, sA, B, sB, l, sl, r = exponential_limit_fit(wavelength_546_V, wavelength_546_d)
    lim                    = find_limit_asymptote(wavelength_546_V, wavelength_546_d, tolerance=1)

    sp    = find_stopping_potential(wavelength_546_V, wavelength_546_d, limtol=1)
    sp_mu = np.mean(sp)
    sp_s  = np.std(sp)

    x     = np.linspace(0, 1.4, 1000)

    plt.figure()
    plt.plot(x, A - B*np.exp(-1*l*x), 'b--', label='Current')

    if lim is not np.zeros(6):
        plt.plot(x, lim[0]*x + lim[1], 'g--', label='Limit')

    plt.errorbar(wavelength_546_V, wavelength_546_d, yerr=d_deflection, fmt='r.', ecolor='k', alpha=0.4)
    plt.errorbar(sp_mu, A - B*np.exp(-1*l*sp_mu), xerr=sp_s, yerr=lim[2], fmt='go', ecolor='k', label=('%1.3f $\pm$ %1.3e$V$' % (sp_mu, sp_s)))

    plt.xlabel('Voltage ($V$)')
    plt.ylabel('Deflection ($mm$)')
    plt.legend(loc='lower right')

    plt.annotate('$f(x)=A + Be^{-\\lambda x}$\n$A=$%f $\pm$ %f\n$B=$%f $\pm$ %f\n$\\lambda=$%f $\pm$ %f' % (A, sA, B, sB, l, sl), xy=(0.8, 1), xytext=(0.8, -15), arrowprops=dict(facecolor='black', headwidth=6, width=.2, shrink=0.05))
    plt.annotate('$y=mx + b$\n$m=$%f$\pm$%f\n$b=$%f$\pm$%f\n$r=$%f' % (lim[0], lim[3], lim[1], lim[4], lim[5]), xy=(0.2, 1), xytext=(0.25, -15), arrowprops=dict(facecolor='black', headwidth=6, width=.2, shrink=0.05))

def plot_577():
    np.seterr(invalid='warn')

    A, sA, B, sB, l, sl, r = exponential_limit_fit(wavelength_577_V, wavelength_577_d)
    lim                    = find_limit_asymptote(wavelength_577_V, wavelength_577_d, tolerance=1)

    sp    = find_stopping_potential(wavelength_577_V, wavelength_577_d, limtol=1, tolerance=[1, 0.75, 0.5])
    sp_mu = np.mean(sp)
    sp_s  = np.std(sp)

    x     = np.linspace(0, 1.0, 1000)

    plt.figure()
    plt.plot(x, A - B*np.exp(-1*l*x), 'b--', label='Current')

    if lim is not np.zeros(6):
        plt.plot(x, lim[0]*x + lim[1], 'g--', label='Limit')

    plt.errorbar(wavelength_577_V, wavelength_577_d, yerr=d_deflection, fmt='r.', ecolor='k', alpha=0.4)
    plt.errorbar(sp_mu, A - B*np.exp(-1.*l*sp_mu), xerr=sp_s, yerr=lim[2], fmt='go', ecolor='k', label=('%1.3f $\pm$ %1.3e$V$' % (sp_mu, sp_s)))

    plt.xlabel('Voltage ($V$)')
    plt.ylabel('Deflection ($mm$)')
    plt.legend(loc='lower right')

    plt.annotate('$f(x)=A + Be^{-\\lambda x}$\n$A=$%f $\pm$ %f\n$B=$%f $\pm$ %f\n$\\lambda=$%f $\pm$ %f' % (A, sA, B, sB, l, sl), xy=(0.85, .12), xytext=(0.63, -3.35), arrowprops=dict(facecolor='black', headwidth=6, width=.2, shrink=0.05))


def plot_voltage():
    l_4358       = c.value / 435.8e-9 * np.ones(3)
    l_546        = c.value / 546e-9 * np.ones(3)
    l_577        = c.value / 577e-9 * np.ones(3)

    voltage_4358 = find_stopping_potential(wavelength_4358_V, wavelength_4358_d)
    voltage_546  = find_stopping_potential(wavelength_546_V, wavelength_546_d, limtol=1)
    voltage_577  = find_stopping_potential(wavelength_577_V, wavelength_577_d, limtol=1, tolerance=[1, 0.75, 0.5])

    wav  = np.concatenate((l_4358, l_546, l_577), axis=0)
    volt = np.concatenate((voltage_4358, voltage_546, voltage_577), axis=0)

    m, b, sy, sm, sb, r = lsq(wav, volt)

    x = np.linspace(wav[0], wav[-1], 1000)

    plt.figure()
    plt.plot(x, m*x + b, 'b--')

    plt.errorbar(l_4358, voltage_4358, yerr=np.std(voltage_4358), fmt='r.', ecolor='k')
    plt.errorbar(l_546, voltage_546, yerr=np.std(voltage_546), fmt='r.', ecolor='k')
    plt.errorbar(l_577, voltage_577, yerr=np.std(voltage_577), fmt='r.', ecolor='k')

    plt.xlabel('Frequency ($Hz$)')
    plt.ylabel('Stopping Voltage ($V$)')

    plt.annotate('$y=mx + b$\n$m=$%1.3e$\pm$%1.3e\n$b=$%f$\pm$%f\n$r=$%f' % (m, sm, b, sb, r), xy=(6*(10**14), .88), xytext=(5.4*(10**14), 1), arrowprops=dict(facecolor='black', headwidth=6, width=.2, shrink=0.05))

def get_h_value():
    l_4358       = c.value / 435.8e-9 * np.ones(3)
    l_546        = c.value / 546e-9 * np.ones(3)
    l_577        = c.value / 577e-9 * np.ones(3)

    voltage_4358 = find_stopping_potential(wavelength_4358_V, wavelength_4358_d)
    voltage_546  = find_stopping_potential(wavelength_546_V, wavelength_546_d, limtol=1)
    voltage_577  = find_stopping_potential(wavelength_577_V, wavelength_577_d, limtol=1, tolerance=[1, 0.75, 0.5])

    wav  = np.concatenate((l_4358, l_546, l_577), axis=0)
    volt = np.concatenate((voltage_4358, voltage_546, voltage_577), axis=0)

    m, b, sy, sm, sb, r = lsq(wav, volt)

    return ('%1.4e ± %1.4e' % (m*q_e.value, sm*q_e.value))

def plot_h_value():
    l_4358       = c.value / 435.8e-9 * np.ones(3)
    l_546        = c.value / 546e-9 * np.ones(3)
    l_577        = c.value / 577e-9 * np.ones(3)

    voltage_4358 = find_stopping_potential(wavelength_4358_V, wavelength_4358_d)
    voltage_546  = find_stopping_potential(wavelength_546_V, wavelength_546_d, limtol=1)
    voltage_577  = find_stopping_potential(wavelength_577_V, wavelength_577_d, limtol=1, tolerance=[1, 0.75, 0.5])

    wav  = np.concatenate((l_4358, l_546, l_577), axis=0)
    volt = np.concatenate((voltage_4358, voltage_546, voltage_577), axis=0)


    m, b, sy, sm, sb, r = lsq(wav, volt)
    he = m * q_e.value # h estimate
    sh = sm * q_e.value

    x = np.linspace(he - 4*sh, he + 4*sh, 1000)

    plt.figure()
    plt.plot(x, mlab.normpdf(x, he, sh), 'b-', label=('$\\bar{h}$=%1.3e $\pm$ %1.3e $J\cdot s$' % (he, sh)))
    plt.axvline(h.value, ls='--', color='k', label=('$h$=%1.3e $J\cdot s$' % (h.value)))
    plt.legend(loc='upper right')
