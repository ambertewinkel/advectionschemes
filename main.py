# Main code with various numerical schemes to solve the advection equation
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import schemes as sch
import experiments as epm
import utils as ut
import analytic as an

def main():
    """
    This function computes and plots the results of various numerical schemes 
    with 1D periodic space and time. Results are compared to the analytic soln. 
    Two initial conditions are considered: a Gaussian distribution and a step 
    function, both defined on a subdomain. 
    Schemes included: FTBS, FTFS, FTCS, CTBS, CTFS, CTCS, Upwind, BTBS, BTCS, MPDATA
    """
    
    # Initial conditions
    nx = 40                     # number of points in space
    nt = 30                     # number of time steps
    xmin, xmax = 0.0, 1.0       # physical domain parameters
    x = np.linspace(xmin, xmax, nx, endpoint=False) # points in space
    dx = x[1] - x[0]            # length of spatial step
    c = np.full(len(x), 0.4)    # Courant number
    dt = 0.1                    # time step
    u = c*dx/dt                 # velocity
    niter = 1                   # number of iterations (for Jacobi or Gauss-Seidel)

    # Calculate initial functions
    psi1_in = an.analytic1(x)
    psi2_in = an.analytic2(x)

    # Calculate analytic solutions
    psi1_an = an.analytic1(x, nt, c)
    psi2_an = an.analytic2(x, nt, c)

    #################
    #### Schemes ####
    #################

    basicschemes = []#['FTBS', 'FTFS', 'FTCS', 'CTBS', 'CTFS', 'CTCS', 'Upwind']
    advancedschemes = ['BTBS', 'BTBS_Jacobi', 'BTBS_GaussSeidel'] #['BTBS', 'BTBS_Jacobi', 'BTBS_GaussSeidel', 'BTCS', 'BTCS_Jacobi', 'BTCS_GaussSeidel', 'MPDATA']
    markers_as = ['x', '', '']
    linestyle_as = ['-','-','--']
    allschemes = basicschemes + advancedschemes
    
    # Calculate numerical results
    for s in allschemes:
        fn = getattr(sch, f'{s}')
        if 'Jacobi' in s or 'GaussSeidel' in s:
            print(s)
            locals()[f'psi1_{s}'] = fn(psi1_in.copy(), nt, c, niter)
            locals()[f'psi2_{s}'] = fn(psi2_in.copy(), nt, c, niter)
        else:
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
    ut.design_figure('Psi1_bs.jpg', f'$\Psi_1$ at t={nt*dt} - Basic Schemes', \
                     'x', '$\Psi_1$', True, -0.5, 1.5)

    plt.plot(x, psi1_an, label='Analytic', linestyle='-', color='k')
    for s in advancedschemes:
        si = advancedschemes.index(s)
        plt.plot(x, locals()[f'psi1_{s}'], label=f'{s}', marker=markers_as[si], linestyle=linestyle_as[si])
    ut.design_figure('Psi1_as.jpg', f'$\Psi_1$ at t={nt*dt} - Advanced Schemes', \
                     'x', '$\Psi_1$', True, -0.5, 1.5)

    plt.plot(x, psi2_an, label='Analytic', linestyle='-', color='k')
    for s in basicschemes:
        plt.plot(x, locals()[f'psi2_{s}'], label=f'{s}')
    ut.design_figure('Psi2_bs.jpg', f'$\Psi_2$ at t={nt*dt} - Basic Schemes', \
                     'x', '$\Psi_2$', True, -0.5, 1.5)
    
    plt.plot(x, psi2_an, label='Analytic', linestyle='-', color='k')
    for s in advancedschemes:
        si = advancedschemes.index(s)
        plt.plot(x, locals()[f'psi2_{s}'], label=f'{s}', marker=markers_as[si], linestyle=linestyle_as[si])
    ut.design_figure('Psi2_as.jpg', f'$\Psi_2$ at t={nt*dt} - Advanced Schemes', \
                     'x', '$\Psi_2$', True,  -0.5, 1.5)
    plt.close()

    #####################
    #### Experiments ####
    #####################

    #### Total variation
    for s in allschemes:
        locals()[f'TV_psi1_{s}'] = epm.totalvariation(locals()[f'psi1_{s}'])
        print(f'1 - Total variation at t={nt*dt} - {s} {locals()[f'TV_psi1_{s}']:.2E}')

    for s in allschemes:
        locals()[f'TV_psi2_{s}'] = epm.totalvariation(locals()[f'psi2_{s}'])
        print(f'2 - Total variation at t={nt*dt} - {s} {locals()[f'TV_psi2_{s}']:.2E}')
    
    print()

    #### Conservation
    csv_psi1_analytic = epm.conservation(psi1_in, psi1_an, dx)
    print(f'1 - Total mass gained at t={nt*dt} - Analytic {csv_psi1_analytic:.2E}')    
    for s in allschemes:
        locals()[f'csv_psi1_{s}'] = epm.conservation(psi1_in, locals()[f'psi1_{s}'], dx)
        print(f'1 - Total mass gained at t={nt*dt} - {s} {locals()[f'csv_psi1_{s}']:.2E}')

    csv_psi2_analytic = epm.conservation(psi2_in, psi2_an, dx)
    print(f'2 - Total mass gained at t={nt*dt} - Analytic {csv_psi2_analytic:.2E}')   
    for s in allschemes:
        locals()[f'csv_psi2_{s}'] = epm.conservation(psi2_in, locals()[f'psi2_{s}'], dx)
        print(f'2 - Total mass gained at t={nt*dt} - {s} {locals()[f'csv_psi2_{s}']:.2E}')
    
    #### Error analysis for a single scheme
    scheme = 'Upwind'
    fn = getattr(sch, f'{scheme}')
    nx_arr = np.array([nx*2, nx, nx/2], dtype=int)
    dx_arr = xmax/nx_arr
    dt_arr = c[0]*dx_arr/u[0]   # This assumes a constant c throughout the domain
    nt_arr = np.array(nt*dt/dt_arr, dtype=int)  # nt*dt is total time above
    rmse_arr = np.zeros(len(dx_arr))

    for i in range(len(nx_arr)):
        c_error = np.full(nx_arr[i], c[0])
        x_error = np.linspace(xmin, xmax, nx_arr[i], endpoint=False)
        psi1_in_error = an.analytic1(x_error)
        psi1_Upwind_error = fn(psi1_in_error.copy(), nt_arr[i], c_error)
        psi1_an_error = an.analytic1(x_error, nt_arr[i], c_error)
        rmse_arr[i] = epm.rmse(psi1_an_error, psi1_Upwind_error, dx_arr[i])

    # log-log plot of RMSE
    plt.loglog(dx_arr, rmse_arr, '-x', label=f'{scheme}')
    plt.loglog(dx_arr, dx_arr, color='green', label='O(dx) accurate')
    ut.design_figure(f'loglog_{scheme}.jpg', f'RMSE for {scheme} scheme', 'dx', 'RMSE')
    
if __name__ == "__main__": main()