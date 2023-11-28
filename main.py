# Main code with various numerical schemes to solve the advection equation
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import schemes as sch
import experiments as epm

def initial1(x, a, b):
    """
    This function returns an array from input array x and constants a and b, 
    with the values from the function y = 0.5*(1-cos(2pi(x-a)/(b-a))), in the 
    range of the domain enclosed by a and b. Outside of this region, the array 
    elements are zero.
    --- Input ---
    x   : 1D array of floats, points to calculate the result of the function for
    a   : float, parameter in function and non-zero result left boundary
    b   : float, parameter in function and non-zero result right boundary
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """
    psi = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] >= a and x[i] < b: # define nonzero region
            psi[i] = 0.5*(1 - np.cos(2*np.pi*(x[i]-a)/(b-a)))

    return psi

def initial2(x, a, b):
    """
    This function returns an array from input array x and constants a and b, 
    with output values 1 in the range of the domain enclosed by a and b and 
    outside of this region, 0. This emulates a step function.
    --- Input ---
    x   : 1D array of floats, points to calculate the result of the step 
          function for
    a   : float, parameter in function and non-zero result left boundary
    b   : float, parameter in function and non-zero result right boundary
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """    
    psi = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] >= a and x[i] < b: # define nonzero region
            psi[i] = 1.

    return psi

def main():
    """
    This function computes and plots the results of various numerical schemes 
    with 1D periodic space and time. Results are compared to the analytic soln. 
    Two initial conditions are considered: a Gaussian distribution and a step 
    function, both defined on a subdomain. 
    Schemes included: CTCS, MPDATA
    """
    
    # Initial conditions
    a, b = 0.1, 0.5
    C = 0.4                     # Courant number
    nx = 40                     # number of points in space
    nt = 10                     # number of time steps
    x = np.linspace(0, 1, nx+1) # points in space
    dx = x[1] - x[0]            # length of spatial step
    dt = 0.1                    # time step
    u = C*dx/dt                 # velocity

    # Calculate initial functions
    psi1_in = initial1(x, a, b)
    psi2_in = initial2(x, a, b)

    # Calculate analytic solutions
    psi1_an = initial1(x - u*nt*dt, a, b)
    psi2_an = initial2(x - u*nt*dt, a, b)

    #################
    #### Schemes ####
    #################

    # CTCS
    psi1_ctcs = sch.ctcs(psi1_in.copy(), nt, C) 
    psi2_ctcs = sch.ctcs(psi2_in.copy(), nt, C) 

    # MPDATA
    psi1_mpdata = sch.mpdata(psi1_in.copy(), nt, C)
    psi2_mpdata = sch.mpdata(psi2_in.copy(), nt, C)

    ##########################
    #### Plotting schemes ####
    ##########################
    
    plt.plot(x, psi1_an, label='Analytic', linestyle='-', color='k')
    plt.plot(x, psi1_ctcs, label='CTCS')
    plt.plot(x, psi1_mpdata, label='MPDATA')
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('$\Psi_1$')
    plt.title('$\Psi_1$ at t=1.0')
    plt.legend()
    plt.savefig('Psi1.jpg')
    plt.show()

    plt.plot(x, psi2_an, label='Analytic', linestyle='-', color='k')
    plt.plot(x, psi2_ctcs, label='CTCS')
    plt.plot(x, psi2_mpdata, label='MPDATA')
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('$\Psi_2$')
    plt.title('$\Psi_2$ at t=1.0')
    plt.legend()
    plt.savefig('Psi2.jpg')
    plt.show()
    
    #####################
    #### Experiments ####
    #####################

    # Total variation
    TV_ctcs = epm.totalvariation(psi1_ctcs)
    TV_mpdata = epm.totalvariation(psi1_mpdata)
    print('1 - Total variation - CTCS', TV_ctcs)
    print('1 - Total variation - MPDATA', TV_mpdata)

    TV_ctcs = epm.totalvariation(psi2_ctcs)
    TV_mpdata = epm.totalvariation(psi2_mpdata)
    print('2 - Total variation - CTCS', TV_ctcs)
    print('2 - Total variation - MPDATA', TV_mpdata)

if __name__ == "__main__": main()