import numpy as np
import copy
import pandas as pd
from scipy import constants
from astropy import units as u

# definition of the constants
# G  = 6.6743e-11 # m^3 / (kg*s^2)
# c  = 299792458  # m / s
# M_sun = 1.98847e30 # kg (solar mass)

# G = 4.3009125e-3 # pc * M_sun^-1 * km/s
# c = 9.7156e-9 # pc / s

G=constants.G*((u.m**3)/(u.kg*u.s**2))
G=(G.to((u.R_sun**3)/(u.M_sun*(u.year*1e6)**2))).value
c=constants.c*(u.m/u.s)
c=c.to(u.R_sun/(u.year*1e6)).value

N = 2 # number of equations to be solved

# set tolerance, i.e. the maximum difference between
# the y values (the difference should be constant theoretically )
tol = .02

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
    dydx = np.zeros(shape=N)
    a, e = yin
    dydx[0] = -64/5   *     G**3*M*m*(M+m)/(c**5*a**3*(1-e**2)**(7/2)) * (1 +  73/ 24*e**2 + 37/96*e**4)
    dydx[1] = -304/15 * e * G**3*M*m*(M+m)/(c**5*a**4*(1-e**2)**(5/2)) * (1 + 121/304*e**2             )

    return dydx

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

    # definitions
    yout, dydx, yt, k1, k2, k3, k4 = (np.zeros(shape=N) for i in range(7))

    # t_i + delta/2
    hh = .5 * h
    xh = xin + hh

    # first step RK
    dydx = deriv( xin, yin, M, m )
    for i in range(N):
        k1[ i ] = h              * dydx[ i ] # f( Y1, ti ) * delta/2
        yt[ i ] = yin[ i ] + 0.5 *   k1[ i ] # Y2

    # second step RK
    dydx = deriv( xh, yt, M, m )
    for i in range(N):
        k2[ i ] = h              * dydx[ i ] # f( Y2, ti + delta/2 ) * delta
        yt[ i ] = yin[ i ] + 0.5 *   k2[ i ] # Y3

    # third step RK
    dydx = deriv( xh, yt, M, m )
    for i in range(N):
        k3[ i ] = h              * dydx[ i ] # f( Y2, ti + delta/2 ) * delta
        yt[ i ] = yin[ i ] +         k3[ i ] # Y3

    # fourth step RK
    dydx = deriv( xin, yt, M, m )
    for i in range(N):
        k4  [ i ] = h * dydx[ i ]
        yout[ i ] = yin[ i ]  + k1 [ i ] / 6. + k2 [ i ] / 3. \
                              + k3 [ i ] / 3. + k4 [ i ] / 6.

    return yout


def ODE_EU( xin, yin, h, M, m ):
    '''
    Euler Method to solve the equations

    Inputs:
        - xin:  variable in which we want to derive (time)
        - yin:  array with the values of a and e
        - h:    spacing in the grid of time
        - M, m: masses of the two black holes

    Output:
        - array with the results of the Euler method
    '''

    # definitions
    yout, dydx = (np.zeros(shape=N) for i in range(2))

    # compute f(tn,yn)
    dydx = deriv( xin, yin, M, m )
    for i in range(N):
        yout[ i ] = yin[ i ] + h * dydx[ i ] # yn + h * f(tn,yn)

    return yout


def delay_time(row, function, h, t):
    '''
    Function to compute the delay time for each entry

    Inputs:
        - row:  row of the BHBH dataframe (it should have the following
                columns: Semimajor, Eccentricity, Mass_0, Mass_1)
        - function: integration function to use (ODE_RK and ODE_EU)
        - h:    spacing in the grid of time
        - t:    array of times (not used, to be removed)
    '''

    # assign the masses
    M1 = row.Mass_0
    M2 = row.Mass_1

    # schwarzschild radius (3 times)
    r_sc = 6 * G * max(M1, M2) / c**2

    # assign the initial values
    a = row.Semimajor
    e = row.Eccentricity

    while a > r_sc:
        a_new, e_new = function( t, (a, e), h, M2, M1 )

        if abs( a_new - a )/a < (0.1*tol): #set adaptive timestep
            h *= 2
            a_new, e_new = function( t, (a, e), h, M2, M1 )

        elif abs(a_new - a)/a > tol:
            while abs(a_new - a)/a > tol:
                h /= 10.
                a_new, e_new = function( t, (a, e), h, M2, M1 )

        a, e = (a_new, e_new)
        t += h

    return pd.Series([t, e])


def analyse(row, function, h, t):
    h_list=[h]
    t_list=[t]

    # assign the masses
    M1 = row.Mass_0
    M2 = row.Mass_1

    # schwarzschild radius (3 times)
    r_sc = 6 * G * max(M1, M2) / c**2

    # assign the initial values
    a = row.Semimajor
    e = row.Eccentricity

    sm_list=[row.Semimajor]
    ec_list=[row.Eccentricity]

    while a > r_sc:
        a_new, e_new = function( t, (a, e), h, M2, M1 )

        if abs( a_new - a )/a < (0.1*tol): #set adaptive timestep
            h *= 2
            a_new, e_new = function( t, (a, e), h, M2, M1 )

        elif abs(a_new - a)/a > tol:
            while abs(a_new - a)/a > tol:
                h /= 10.
                a_new, e_new = function( t, (a, e), h, M2, M1 )

        a, e = (a_new, e_new)
        t += h
        h_list.append(h)
        t_list.append(t)
        sm_list.append(a_new)
        ec_list.append(e_new) 

    return pd.Series([t_list, h_list, sm_list, ec_list, t])
