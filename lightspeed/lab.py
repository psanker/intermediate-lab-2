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

def maxima(f, xmin, xmax, precision=1e-5):
    '''
    Numerically find the maxima on a domain [xmin, xmax]

    Returns array of all x values which are maxima for f
    '''

    N    = int(1. / precision)

    x    = np.linspace(xmin, xmax, N)
    dx   = x[1] - x[0]
    maxs = []

    for i in range(len(x)):
        # Reject endpoints
        if i == 0:
            continue
        elif i == len(x) - 1:
            break

        sd = (f(x[i + 1]) - (2. * f(x[i])) + f(x[i - 1])) / dx

        if sd < 0.:
            if f(x[i - 1]) < f(x[i]) and f(x[i + 1]) < f(x[i]):
                maxs.append(x[i])
                continue

            # Now for some derivative magic
            d_next = (f(x[i + 1]) - f(x[i])) / dx
            d_last = (f(x[i]) - f(x[i - 1])) / dx

            d_avg  = (d_next + d_last) / 2.

            if d_avg == 0.0 or d_next == 0.0:
                maxs.append(x[i])
            elif d_next < 0.0 and d_last > 0.0:
                # weighted avg based on the slope
                # The flatter the slope, the more weight it should have
                total = np.abs(1. / d_last) + np.abs(1. / d_next)
                a_l   = np.abs(1. / d_last) / total
                a_n   = np.abs(1. / d_next) / total

                avg1  = (x[i - 1] + x[i]) / 2.
                avg2  = (x[i + 1] + x[i]) / 2.

                avg = (a_l * avg1) + (a_n * avg2)

                maxs.append(avg)

    return np.array(maxs)

def mc_sample(N, f, fmax, xmin=-5, xmax=5):
    '''
    Find N values that fall under the function 'f'
    '''

    ret = []

    while len(ret) < N:
        x, y = (np.random.uniform(low=xmin, high=xmax), np.random.uniform(low=0.0, high=fmax))

        if y <= f(x):
            ret.append(x)

    return np.array(ret)

def mc_integrate(N, f, fmax, xmin, xmax):
    '''
    Monte-Carlo integrator for distribution functions
    '''

    tot_area = fmax * (xmax - xmin)
    count    = 0

    for i in range(N):
        x, y = (np.random.uniform(low=xmin, high=xmax), np.random.uniform(low=0.0, high=fmax))

        if y <= f(x):
            count += 1

    return tot_area * (float(count) / float(N))

#############################################################
# 4. Data
#############################################################

source_dist = .095 #meters
mirror_dist = .05 #meters
xtra = source_dist - mirror_dist
temp = 20.0 #celsius

air_x = (np.array([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]) * 2. * 1e-2) + xtra
air_t = np.array([0, 36.0, 96.0, 176.0, 236.0, 316.0, 400.0, 436.0, 536.0, 616.0]) * 1e-12 # seconds
xi = 56.5 * 1e-2 + xtra
water_xf = np.array([89.0, 87.3, 86.4, 86.8]) * 1e-2 + xtra
poly_xf = np.array([78.0, 77.9, 79.0, 77.9]) * 1e-2 + xtra
glass_xf = np.array([84.0, 83.6, 86.7, 85.3]) * 1e-2 + xtra
unknown_xf = np.array([106.3, 109.9, 107.7, 107.8]) * 1e-2 + xtra
disterr = np.sqrt((.002/(air_x - xtra))**2 + (.002/.095)**2 + (.01/.5)**2) * air_x
timeerr = 5.7 * 1e-12

#############################################################
# 5. Lab-specific functions
#############################################################

def bimodal(x):
    z   = 2. * np.sqrt(2. * PI)

    val = np.exp((-(x - 1.)**2.) / 2.) + np.exp((-(x + 1.)**2.) / 2.)

    return (1. / z) * val

# Test function to see at what point things would get screwy
def bimodal2(x):
    s   = 5.
    z   = 10. * np.sqrt(2. * PI)

    val = np.exp((-(x - 10.)**2.) / (2. * s**2.)) + np.exp((-(x + 10.)**2.) / (2. * s**2.))

    return (1. / z) * val

def find_refraction(x, l):
    n  = (np.mean(x)-xi + l) / l
    sn = np.sqrt((np.std(x)/np.mean(x))**2 + (.001/.565)**2)*n
    return n, sn

def get_dummy():
    return maxima(bimodal2, xmin=-20, xmax=20, precision=5e-5)

def get_refraction():
    w, sw = find_refraction(water_xf, .5) #accepted value is 1.3, ours is 1.62
    p, sp = find_refraction(poly_xf, .5) #accepted value is 1.4, ours is 1.43
    g, sg = find_refraction(glass_xf, .5) #accepted value is 1.49, ours is 1.57
    u, su = find_refraction(unknown_xf, .2) #something here is very wrong, the value is 3.57
    return ('Water: %1.3f ± %1.3f\nPoly: %1.3f ± %1.3f\nGlass: %1.3f ± %1.3f\nUnknown %1.3f ± %1.3f' % (w, sw, p, sp, g, sg, u, su))

def get_speeds():
    c, b, sy, sc, sb, r = lsq(air_t, air_x)
    dc = np.sqrt(np.mean((sc/c)**2) + np.mean((disterr/air_x)**2) + np.mean((timeerr/air_t[1:])**2))*c
    w, sw = find_refraction(water_xf, .5)
    p, sp = find_refraction(poly_xf, .5)
    g, sg = find_refraction(glass_xf, .5)
    u, su = find_refraction(unknown_xf, .2)
    wspeed  = c / w
    swspeed = np.sqrt((sw/w)**2 + (dc/c)**2)*wspeed
    pspeed  = c / p
    spspeed = np.sqrt((sp/p)**2 + (dc/c)**2)*pspeed
    gspeed  = c / g
    sgspeed = np.sqrt((sg/g)**2 + (dc/c)**2)*gspeed
    uspeed  = c / u
    suspeed = np.sqrt((su/u)**2 + (dc/c)**2)*uspeed

    return ('Air: %1.f ± %1.f\nWater: %1.f ± %1.f\nPoly: %1.f ± %1.f\nGlass: %1.f ± %1.f\nUnknown %1.f ± %1.f' % (c, dc, wspeed, swspeed, pspeed, spspeed, gspeed, sgspeed, uspeed, suspeed))

def plot_airspeed():
    x = np.linspace(0, 616*1e-12, 1000)

    m, b, sy, sm, sb, r = lsq(air_t, air_x)
    plt.figure()
    plt.plot(air_t, air_x, 'r.', label='Air Data')
    plt.plot(x, m*x+b, 'b-', label='Linear Fit')
    plt.annotate('$y=mx + b$\n$m=$%1.3e$\pm$%1.3e\n$b=$%1.3e$\pm$%1.3e\n$r=$%1.3f' % (m, sm, b, sb, r), xy=(2.9e-10, 2.15), xytext=(3.1e-10, 2.08), arrowprops=dict(facecolor='black', headwidth=6, width=.2, shrink=0.05))
    plt.xlabel('Time Delay (s)')
    plt.ylabel('Path Length (m)')
    plt.legend(loc='upper left')

def plot_anglesim():
    # bimodal max val -- numerically computed
    bmax = 1. / np.sqrt(2. * PI * np.exp(1.))

    # degrees to radians conversion factor
    conv = PI / 180.

    # Monte-Carlo sample with bimodal kernel function
    samp = mc_sample(1000, bimodal, bmax, xmin=-10, xmax=10)

    time = []
    dis  = []

    for i in range(len(air_x)):

        for j in range(len(samp)):
            # Test length based on angular deflection
            l = air_x[i] * np.cos(conv * samp[j])

            time.append(air_t[i])
            dis.append(l)

    time = np.array(time)
    dis  = np.array(dis)

    m, b, sy, sm, sb, r = lsq(time, dis)
    x = np.linspace(min(time), max(time), 1000)

    plt.figure()
    plt.plot(time, dis, 'r.', label='Bimodal simulation')
    plt.plot(x, m*x + b, 'b-', label='Linear fit')

    plt.annotate('$y = mx + b$\n$m = %1.3e\\pm%1.3e$\n$b = %1.3e\\pm%1.3e$\n$r = %1.4f$' % (m, sm, b, sb, r),
                 xy = (3e-10, 2.16), xytext = (1.e-10, 2.20),
                 arrowprops = dict(
                    facecolor = 'k',
                    headwidth = 6,
                    width  = .2,
                    shrink = 0.05
                 ))

    plt.legend(loc='lower right')
    plt.xlabel('Time ($s$)')
    plt.ylabel('Distance ($m$)')

def plot_anglesim2():
    '''
    Test function to see if strangely wonky bimodal results would reproduce speed of light error
    '''

    # bimodal max val -- numerically computed
    bmax = np.mean(bimodal2(maxima(bimodal2, -11., 11.)))

    # degrees to radians conversion factor
    conv = PI / 180.

    # Monte-Carlo sample with bimodal kernel function
    samp = mc_sample(1000, bimodal2, bmax, xmin=-20, xmax=20)

    time = []
    dis  = []

    for i in range(len(air_x)):

        for j in range(len(samp)):
            # Test length based on angular deflection
            l = air_x[i] * np.cos(conv * samp[j])

            time.append(air_t[i])
            dis.append(l)

    time = np.array(time)
    dis  = np.array(dis)

    m, b, sy, sm, sb, r = lsq(time, dis)
    x = np.linspace(min(time), max(time), 100)

    plt.figure()
    plt.plot(time, dis, 'r.', label='Bimodal simulation')
    plt.plot(x, m*x + b, 'b-', label='Linear fit')

    plt.annotate('$y = mx + b$\n$m = %1.3e\\pm%1.3e$\n$b = %1.3e\\pm%1.3e$\n$r = %1.4f$' % (m, sm, b, sb, r),
                 xy = (3e-10, 2.16), xytext = (1.e-10, 2.20),
                 arrowprops = dict(
                    facecolor = 'k',
                    headwidth = 6,
                    width  = .2,
                    shrink = 0.05
                 ))

    plt.legend(loc='lower right')
    plt.xlabel('Time ($s$)')
    plt.ylabel('Distance ($m$)')
