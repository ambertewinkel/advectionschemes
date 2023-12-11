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
    nx = 40                     # number of points in space
    nt = 100                    # number of time steps
    x = np.linspace(0, 1, nx+1) # points in space
    dx = x[1] - x[0]            # length of spatial step
    c = np.full(len(x), 0.4)    # Courant number
    dt = 0.1                    # time step
    u = c*dx/dt                 # velocity

    # Calculate initial functions
    psi1_in = initial1(x, a, b)
    psi2_in = initial2(x, a, b)

    # Calculate analytic solutions
    psi1_an = initial1(x - u*nt*dt, a, b)
    psi2_an = initial2(x - u*nt*dt, a, b)

    #################
    #### Schemes ####
    #################

    # FTBS
    psi1_ftbs = sch.ftbs(psi1_in.copy(), nt, c) 
    psi2_ftbs = sch.ftbs(psi2_in.copy(), nt, c) 

    # FTFS
    psi1_ftfs = sch.ftfs(psi1_in.copy(), nt, c) 
    psi2_ftfs = sch.ftfs(psi2_in.copy(), nt, c) 

    # FTCS
    psi1_ftcs = sch.ftcs(psi1_in.copy(), nt, c) 
    psi2_ftcs = sch.ftcs(psi2_in.copy(), nt, c) 

    # CTBS
    psi1_ctbs = sch.ctbs(psi1_in.copy(), nt, c) 
    psi2_ctbs = sch.ctbs(psi2_in.copy(), nt, c) 

    # CTFS
    psi1_ctfs = sch.ctfs(psi1_in.copy(), nt, c) 
    psi2_ctfs = sch.ctfs(psi2_in.copy(), nt, c) 

    # CTCS
    psi1_ctcs = sch.ctcs(psi1_in.copy(), nt, c) 
    psi2_ctcs = sch.ctcs(psi2_in.copy(), nt, c) 
    
    # Upwind
    psi1_upwind = sch.upwind(psi1_in.copy(), nt, c) 
    psi2_upwind = sch.upwind(psi2_in.copy(), nt, c) 

    # MPDATA
    psi1_mpdata = sch.mpdata(psi1_in.copy(), nt, c)
    psi2_mpdata = sch.mpdata(psi2_in.copy(), nt, c)

    ##########################
    #### Plotting schemes ####
    ##########################
    
    plt.plot(x, psi1_an, label='Analytic', linestyle='-', color='k')
    plt.plot(x, psi1_ftbs, label='FTBS')
    plt.plot(x, psi1_ftfs, label='FTFS')
    plt.plot(x, psi1_ftcs, label='FTCS')
    plt.plot(x, psi1_ctbs, label='CTBS')
    plt.plot(x, psi1_ctfs, label='CTFS')
    plt.plot(x, psi1_ctcs, label='CTCS')
    plt.plot(x, psi1_upwind, label='Upwind')
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('$\Psi_1$')
    plt.title(f'$\Psi_1$ at t={nt*dt} - Basic Schemes - c = {c}')
    plt.legend()
    plt.savefig('Psi1_bs.jpg')
    plt.show()

    plt.plot(x, psi1_an, label='Analytic', linestyle='-', color='k')
    plt.plot(x, psi1_mpdata, label='MPDATA')
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('$\Psi_1$')
    plt.title(f'$\Psi_1$ at t={nt*dt} - Alternative Schemes - c = {c}')
    plt.legend()
    plt.savefig('Psi1_as.jpg')
    plt.show()

    plt.plot(x, psi2_an, label='Analytic', linestyle='-', color='k')
    plt.plot(x, psi2_ftbs, label='FTBS')
    plt.plot(x, psi2_ftfs, label='FTFS')
    plt.plot(x, psi2_ftcs, label='FTCS')
    plt.plot(x, psi2_ctbs, label='CTBS')
    plt.plot(x, psi2_ctfs, label='CTFS')
    plt.plot(x, psi2_ctcs, label='CTCS')
    plt.plot(x, psi2_upwind, label='Upwind')
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('$\Psi_2$')
    plt.title(f'$\Psi_2$ at t={nt*dt} - Basic Schemes - c = {c}')
    plt.legend()
    plt.savefig('Psi2_bs.jpg')
    plt.show()
    
    plt.plot(x, psi2_an, label='Analytic', linestyle='-', color='k')
    plt.plot(x, psi2_mpdata, label='MPDATA')
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('$\Psi_2$')
    plt.title(f'$\Psi_2$ at t={nt*dt} - Alternative Schemes - c = {c}')
    plt.legend()
    plt.savefig('Psi2_as.jpg')
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