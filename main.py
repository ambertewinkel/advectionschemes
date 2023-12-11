# Main code with various numerical schemes to solve the advection equation
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import schemes as sch
import experiments as epm

def analytic1(x, nt=0., c=0.):
    """
    This function returns an array from input array x and constants a and b advected 
    by velocity u for a time t. The initial condition has values from the function 
    y = 0.5*(1-cos(2pi(x-a)/(b-a))), in the range of the domain enclosed by a and b. 
    Outside of this region, the array elements are zero.
    --- Input ---
    x   : 1D array of floats, points to calculate the result of the function for
    nt  : integer, number of time steps advected
    c   : float or 1D array of floats, Courant number for advection (c = u*dt/dx)
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """
    a, b = 0.1, 0.5
    psi = np.zeros(len(x))
    dx = x[1] - x[0]
    xmax = x[-1] + dx       # size of domain (assuming periodicity)
    x0 = (x - c*nt*dx)%xmax # initial x that input x corresponds to after advection (u*t = c*nt*dx)
    for i in range(len(x)):
        if x0[i] >= a and x0[i] < b: # define nonzero region
            psi[i] = 0.5*(1 - np.cos(2*np.pi*(x0[i]-a)/(b-a)))

    return psi

def analytic2(x, nt=0., c=0.):
    """
    This function returns an array from input array x and constants a and b advected 
    by velocity u for a time t. The initial condition has output values 1 in the range
    of the domain enclosed by a and b and outside of this region, 0. This emulates a 
    step function. 
    --- Input ---
    x   : 1D array of floats, points to calculate the result of the step 
        function for
    nt  : integer, number of time steps advected (nt = t/dt)
    c   : float or 1D array of floats, Courant number for advection (c = u*dt/dx)
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """    
    a, b = 0.1, 0.5
    psi = np.zeros(len(x))
    dx = x[1] - x[0]
    xmax = x[-1] + dx       # size of domain (assuming periodicity)
    x0 = (x - c*nt*dx)%xmax # initial x that input x corresponds to after advection (u*t = c*nt*dx)
    for i in range(len(x)):
        if x0[i] >= a + 1.E-6 and x0[i] < b - 1.E-6: # define nonzero region
            psi[i] = 1.

    return psi

def main():
    """
    This function computes and plots the results of various numerical schemes 
    with 1D periodic space and time. Results are compared to the analytic soln. 
    Two initial conditions are considered: a Gaussian distribution and a step 
    function, both defined on a subdomain. 
    Schemes included: FTBS, FTFS, FTCS, CTBS, CTFS, CTCS, Upwind, MPDATA
    """
    
    # Initial conditions
    nx = 40                     # number of points in space
    nt = 30                     # number of time steps
    x = np.linspace(0, 1, nx, endpoint=False) # points in space
    dx = x[1] - x[0]            # length of spatial step
    c = np.full(len(x), 0.4)    # Courant number
    dt = 0.1                    # time step
    u = c*dx/dt                 # velocity

    # Calculate initial functions
    psi1_in = analytic1(x)
    psi2_in = analytic2(x)

    # Calculate analytic solutions
    psi1_an = analytic1(x, nt, c)
    psi2_an = analytic2(x, nt, c)

    #################
    #### Schemes ####
    #################

    basicschemes = ['FTBS', 'FTFS', 'FTCS', 'CTBS', 'CTFS', 'CTCS', 'Upwind']
    advancedschemes = ['MPDATA']
    allschemes = basicschemes + advancedschemes
    
    # Calculate numerical results
    for s in allschemes:
        fn = getattr(sch, f'{s}')
        locals()[f'psi1_{s}'] = fn(psi1_in.copy(), nt, c)
        locals()[f'psi2_{s}'] = fn(psi2_in.copy(), nt, c)
    
    # Print Courant numbers
    print('The Courant numbers are:', c)
    
    ##########################
    #### Plotting schemes ####
    ##########################
    
    plt.plot(x, psi1_an, label='Analytic', linestyle='-', color='k')
    for s in basicschemes:
        plt.plot(x, locals()[f'psi1_{s}'], label=f'{s}')
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('$\Psi_1$')
    plt.title(f'$\Psi_1$ at t={nt*dt} - Basic Schemes')
    plt.legend()
    plt.savefig('Psi1_bs.jpg')
    plt.show()   

    plt.plot(x, psi1_an, label='Analytic', linestyle='-', color='k')
    for s in advancedschemes:
        plt.plot(x, locals()[f'psi1_{s}'], label=f'{s}')
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('$\Psi_1$')
    plt.title(f'$\Psi_1$ at t={nt*dt} - Advanced Schemes')
    plt.legend()
    plt.savefig('Psi1_as.jpg')
    plt.show()

    plt.plot(x, psi2_an, label='Analytic', linestyle='-', color='k')
    for s in basicschemes:
        plt.plot(x, locals()[f'psi2_{s}'], label=f'{s}')
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('$\Psi_2$')
    plt.title(f'$\Psi_2$ at t={nt*dt} - Basic Schemes')
    plt.legend()
    plt.savefig('Psi2_bs.jpg')
    plt.show()
    
    plt.plot(x, psi2_an, label='Analytic', linestyle='-', color='k')
    for s in advancedschemes:
        plt.plot(x, locals()[f'psi2_{s}'], label=f'{s}')
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('$\Psi_2$')
    plt.title(f'$\Psi_2$ at t={nt*dt} - Advanced Schemes')
    plt.legend()
    plt.savefig('Psi2_as.jpg')
    plt.show()

    #####################
    #### Experiments ####
    #####################

    # Total variation
    for s in allschemes:
        locals()[f'TV_psi1_{s}'] = epm.totalvariation(locals()[f'psi1_{s}'])
        print(f'1 - Total variation at t={nt*dt} - {s}', locals()[f'TV_psi1_{s}'])

    for s in allschemes:
        locals()[f'TV_psi2_{s}'] = epm.totalvariation(locals()[f'psi2_{s}'])
        print(f'2 - Total variation at t={nt*dt} - {s}', locals()[f'TV_psi2_{s}'])

if __name__ == "__main__": main()