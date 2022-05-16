import numpy as np
import copy
import pandas as pd

from dask import compute
from dask import delayed

# definition of the constants
#G  = 6.6743e-11 # m^3 / (kg*s^2)
#c  = 299792458  # m / s
#M_sun = 1.98847e30 # kg (solar mass)

#G = 4.3009125e-3 # pc * M_sun^-1 * km/s
#c = 9.7156e-9 # pc / s
G = c = 1
N = 2 # number of equations to be solved

@delayed
def dadt( a, e, M, m ):
    return -64/5 * G**3*M*m*(M+m)/(c**5*a**3*(1-e**2)**(7/2)) * (1 +  73/ 24*e**2 + 37/96*e**4)

@delayed
def dedt( a, e, M, m ):
    return -304/15 * e * G**3*M*m*(M+m)/(c**5*a**4*(1-e**2)**(5/2)) * (1 + 121/304*e**2 )

def deriv( xin, yin, M, m ):
    '''
    Computes the derivatives

    Inputs:
        - xin:  variable in which we want to derive
        - yin:  array with the values of a and e
        - M, m: masses of the two black holes

    Output:
        - array with the computed derivatives
    '''
    a, e = yin
    return [dadt(a, e, M, m), dedt(a, e, M, m)]

@delayed(nout=2)
def first_step( xin, yin, h, M, m ):
    yt, k1 = (np.empty(shape=N, dtype=object) for i in range(2))
    dydx = compute(*deriv( xin, yin, M, m ))
    for i in range(N):
        k1[ i ] = h              * dydx[ i ] # f( Y1, ti ) * delta/2
        yt[ i ] = yin[ i ] + 0.5 *   k1[ i ] # Y2
    return yt, k1

@delayed(nout=2)
def second_step( xin, yin, y1, h, M, m ):
    yt, k2 = (np.empty(shape=N, dtype=object) for i in range(2))
    dydx = compute(*deriv( xin, y1, M, m ))
    for i in range(N):
        k2[ i ] = h              * dydx[ i ] # f( Y2, ti + delta/2 ) * delta
        yt[ i ] = yin[ i ] + 0.5 *   k2[ i ] # Y3
    return yt, k2

@delayed(nout=2)
def third_step( xin, yin, y2, h, M, m ):
    yt, k3 = (np.empty(shape=N, dtype=object) for i in range(2))
    dydx = compute(*deriv( xin, y2, M, m ))
    for i in range(N):
        k3[ i ] = h              * dydx[ i ] # f( Y2, ti + delta/2 ) * delta
        yt[ i ] = yin[ i ] +         k3[ i ] # Y3
    return yt, k3

@delayed
def fourth_step( xin, yin, y3, h, M, m, k ):
    yt, k4 = (np.empty(shape=N, dtype=object) for i in range(2))

    dydx = compute(*deriv( xin, y3, M, m ))
    for i in range(N):
        k4  [ i ] = h * dydx[ i ]
        yt[ i ] = yin[ i ]  + k[0][ i ] / 6. + k[1][ i ] / 3. + k[2][ i ] / 3. + k4[ i ] / 6.
    return yt

def ODE_RK( xin, yin, h, M, m ):
    '''
    4-th order Runge-Kutta method

    Inputs:
        - xin:  variable in which we want to derive (time)
        - yin:  array with the values of a and e
        - h:    spacing in the grid of time
        - M, m: masses of the two black holes

    Output:
        - array with the results of the Runge-Kutta method
    '''

    # t_i + delta/2
    hh = .5 * h
    xh = xin + hh

    # first step RK
    yt, k1 = first_step( xin, yin, h, M, m )

    # second step RK
    yt, k2 = second_step( xh, yin, yt, h, M, m )

    # third step RK
    yt, k3 = third_step( xh, yin, yt, h, M, m )

    # fourth step RK
    return fourth_step( xin, yin, yt, h, M, m, (k1, k2, k3))