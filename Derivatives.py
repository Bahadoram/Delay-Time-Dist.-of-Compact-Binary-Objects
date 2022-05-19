import numpy as np
import copy
from scipy import constants
from astropy import units as u

# definition of the constants
#G  = 6.6743e-11 # m^3 / (kg*s^2)
#c  = 299792458  # m / s
#M_sun = 1.98847e30 # kg (solar mass)

#G = 4.3009125e-3 # pc * M_sun^-1 * km/s
#c = 9.7156e-9 # pc / s

G=constants.G*((u.m**3)/u.kg*u.s**2)
G=G.to((u.R_sun**3)/u.M_sun*(u.year*1e6)**2).value
c=constants.c*u.m/u.s
c=c.to(u.R_sun/u.year*1e6).value

N = 2 # number of equations to be solved


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

    # conversion in standard units
    #M *= M_sun
    #m *= M_sun

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
        yout[ i ] = yin[ i ]  + k1 [ i ] / 6. + k2 [ i ] / 3. + k3 [ i ] / 3. + k4 [ i ] / 6.

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
