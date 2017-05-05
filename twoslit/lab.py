# -*- coding: utf-8 -*-

#############################################################
# 1. Imports
#############################################################

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math
import numba

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

def mc_integrate(f, xmin, xmax, N=10000, seed=-1):
    '''
    Monte-Carlo integrator for distribution functions
    '''

    if seed != -1:
        np.random.seed(seed)

    samples = np.random.uniform(low=xmin, high=xmax, size=N)
    values  = np.empty(N)

    R  = xmax - xmin
    dx = R / float(N)

    values = f(samples)
    s_int  = (R / np.sqrt(N)) * np.std(values, ddof=1)

    return np.sum(values)*dx, s_int

#############################################################
# 4. Data
#############################################################

#Laser Source

bg_laser = .007 #background voltage

pos_laser = np.array([3.33, 3.37, 3.39, 3.43, 3.49, 3.7, 3.71, 3.78, 3.8, 3.82, 3.84, 3.85, 3.88, 4.42, 5.1, 5.13, 5.15, 5.16, 5.17, 5.19, 5.21, 5.27, 5.46, 5.53, 5.56, 5.59, 5.61, 5.65]) #mm
intensity_laser = np.array([.008, .05, .107, .248, .34, .342, .349, .608, .813, 1.018, 1.2, 1.318, 1.471, 1.518, 1.473, 1.372, 1.252, 1.138, 1.016, .836, .654, .462, .465, .386, .259, .139, .053, .008]) - bg_laser #V

#Bulb Source

pulse_rand = np.array([1.053, 1.053, 1.072, 1.078, 1.062, .966, 1.084, 1.039, 1.041, 1.039]) #kHz

counts_double = np.array([.379, .468, .444, .367, .241, .269, .486, .714, .881, .738, .434, .163, .220, .530, .953, 1.037, .816, .376, .137, .273, .561, .806, .755, .525, .295, .229, .269, .371, .403, .425, .295]) #kHz
pos_double = np.linspace(3.75, 6.75, num=len(counts_double))

counts_single1 = np.array([.161, .185, .202, .233, .215, .271, .265, .320, .312, .304, .362, .317, .315, .261, .251, .259, .213, .234, .190, .168, .140]) #kHz
counts_single2 = np.array([.154, .181, .193, .240, .222, .282, .268, .277, .314, .311, .349, .306, .314, .278, .259, .265, .211, .238, .183, .156, .145]) #kHz -- so so dirty
pos_single = np.linspace(4.25, 6.25, num=len(counts_single1))

bg_bulb = np.array([.022, .019, .024, .024, .02, .026, .018, .018, .019, .018, .029, .020, .018, .019, .02, .029])

# Analysis info for Patrick
data_a = .29e-3
data_d = 1.45e-3
data_k = TWO_PI / (670e-9)

SEED   = 912837438 # a random-ish number that fixes the integrator to produce self-consistent information

#############################################################
# 5. Lab-specific functions
#############################################################

# Unnormalized probability distribution for double-slit
@numba.jit
def p(t, a=data_a, d=data_d, k=data_k):
    ret = np.zeros(len(t))
    c1 = 1.

    for i in range(len(ret)):
        alp = (k * a * np.sin(t[i])) / 2.

        c2 = (np.sin(alp) / alp)**2.
        c3 = (np.cos((k * d * np.sin(t[i])) / 2.))**2.

        if alp == 0.0:
            ret[i] = 2. * c1 * c3
        else:
            ret[i] = c1 * c2 * c3

    return ret

# Unnormalized probability distribution for single-slit
@numba.jit
def env(t, a=data_a, d=data_d, k=data_k):
    ret = np.zeros(len(t))
    c1 = 1.

    for i in range(len(ret)):
        alp = (k * a * np.sin(t[i])) / 2.

        c2 = (np.sin(alp) / alp)**2.

        if alp == 0.0:
            ret[i] = 2. * c1
        else:
            ret[i] = c1 * c2

    return ret

def norms(size=1000000):
    n1, dn1 = mc_integrate(p, -PI, PI, size, seed=SEED)
    n2, dn2 = mc_integrate(env, -PI, PI, size, seed=SEED)

    return n1, dn1, n2, dn2

def get_normfactors():
    n1, dn1, n2, dn2 = norms()

    # Norm factor for double-slit
    str1 = '%1.3e ± %1.3e' % (n1, dn1)

    # Norm factor for single-slit
    str2 = '%1.3e ± %1.3e' % (n2, dn2)

    # Ratio of normalizing factors i.e. the max amplitudes
    q   = n2 / n1
    s2q = (dn2 / n2)**2 + (dn1 / n1)**2

    str3 = '%1.3f ± %1.3e' % (q, np.sqrt(s2q))

    return 'Double (N1): %s\nSingle (N2): %s\nN1/N2: %s' % (str1, str2, str3)

def get_normcheck():
    n1, dn1, n2, dn2 = norms()

    v1, dv1 = mc_integrate(lambda x: (1 / n1) * p(x), -PI, PI, 1000000, seed=SEED)
    v2, dv2 = mc_integrate(lambda x: (1 / n2) * env(x), -PI, PI, 1000000, seed=SEED)

    str1 = '%1.3f ± %1.3e' % (v1, dv1)
    str2 = '%1.3f ± %1.3e' % (v2, dv2)

    return 'Norm test\n-----\nDouble: %s\nSingle: %s' % (str1, str2)

#Laser Source
def get_singlemaxleft():
    '''
    Estimates the maximum intensity and uncertainty for the left side single slit pattern of the laser

    Returns an array of [Intensity max, Intensity std, Position Max, Position std]
    '''
    li = np.array([.34, .342, .349])
    mu_li = np.mean(li)
    dli = np.std(li)
    lp = np.array([3.49, 3.7, 3.71])
    mu_lp = np.mean(lp)
    dlp = np.std(lp)

    return np.array([mu_li, dli, mu_lp, dlp])

def get_singlemaxright():
    '''
    Estimates the maximum intensity and uncertainty for the right side single slit pattern of the laser

    Returns an array of [Intensity max, Intensity std, Position Max, Position std]
    '''
    ri = np.array([.462, .465])
    mu_ri = np.mean(ri)
    dri = np.std(ri)
    rp = np.array([5.27, 5.46])
    mu_rp = np.mean(rp)
    drp = np.std(rp)

    return np.array([mu_ri, dri, mu_rp, drp])

def get_ratio():
    doub  = get_doublemax()
    singl = get_singlemaxleft()
    singr = get_singlemaxright()

    r = np.array([doub[0] / singl[0], doub[0] / singr[0]])

    return np.mean(r), np.std(r)

def plot_inttheory():
    n1, dn1, n2, dn2 = norms()
    t = np.linspace(-.002*PI, .002*PI, 10000)

    plt.figure()
    plt.plot(t, (1 / n1)*p(t), label='Double Slit', alpha=0.5)
    plt.plot(t, (1 / n2)*env(t), label='Single Slit', color='r')

    plt.xlabel('Angle ($rad$)')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper left')

def get_doublemax():
    '''
    Estimates the maximum intensity and uncertainty for the double slit pattern of the laser

    Returns an array of [Intensity max, Intensity std, Position, Postion uncertainty]
    '''
    i = np.array([1.509, 1.518])
    mu_i = np.mean(i)
    di = np.std(i)
    pos = 4.42
    dp = .01

    return np.array([mu_i, di, pos, dp])


def plot_laser():
    '''
    Plots the Two Slit Intensity Pattern for the laser source
    '''
    leftside = get_singlemaxleft()
    rightside = get_singlemaxright()
    double = get_doublemax()
    Intensitymax = np.array([leftside[0], 1.518, rightside[0]])
    Intensityerr = np.array([leftside[1], double[1], rightside[1]])
    Positionmax = np.array([leftside[2], 4.42, rightside[2]])
    Positionerr = np.array([leftside[3], .01, rightside[3]])


    plt.figure()
    plt.plot(pos_laser, intensity_laser, '.b-', label='Maximum Intensity Distribution')
    plt.errorbar(Positionmax, Intensitymax, xerr=Positionerr, yerr=Intensityerr, fmt='ro', ecolor='k', label='Intensity Maximums')

    plt.legend(loc='lower center')
    plt.xlabel('Blocker Position ($mm$)')
    plt.ylabel('Intensity ($V$)')


#Bulb Source
def get_satisfaction():
    '''
    Answers the question "Are you satisfied that most of your counts lie in the range of the average?"
    '''
    C = np.mean(pulse_rand)
    dC = np.std(pulse_rand)
    bad = []

    for i in range(len(pulse_rand)):
        if pulse_rand[i] < (C - dC):
            bad.append(1)
        elif pulse_rand[i] > (C + dC):
            bad.append(1)
    if len(bad) < 3:
        print("All is well, only %d counts lie outside the range." % len(bad))
    return '%1.3f ± %1.3f kHz' % (C, dC)

def get_background():
    '''
    Determines best value for background counts with the bulb source
    '''
    bg = np.mean(bg_bulb)
    dbg = np.std(bg_bulb)
    return bg, dbg

def plot_bulb():
    '''
    Plots intensity pattern for bulb source
    '''
    BG, dBG = get_background()
    plt.figure()
    plt.plot(pos_double, (counts_double - BG), '.b-', alpha=0.7, label='Two Slit Intensity')
    plt.plot(pos_single, (counts_single1 - BG), '.g-', alpha=0.7, label='Left Single Slit Intensity')
    plt.plot(pos_single, (counts_single2 - BG), '.m-', alpha=0.7, label='Right Single Slit Intensity')
    plt.legend(loc='upper right')
    plt.xlabel('Detector Position ($mm$)')
    plt.ylabel('Intensity ($V$)')
