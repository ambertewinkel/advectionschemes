# Main code with various numerical schemes to solve the advection equation
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import schemes as sch
import experiments as epm
import utils as ut
import analytic as an
import grid as gr

# !!! future: allow the wind to change over time?

def main():
    """
    This function computes and plots the results of various numerical schemes 
    with 1D periodic space and time. Results are compared to the analytic soln. 
    Two initial conditions are considered: a Gaussian distribution and a step 
    function, both defined on a subdomain. 
    Schemes included: FTBS, FTFS, FTCS, CTBS, CTFS, CTCS, Upwind, BTBS, BTFS, BTCS, CNBS, MPDATA
    """
    
    # Initial conditions
    dt = 0.1                    # time step
    nt = 100                    # number of time steps
    nx = 40                     # number of points in space
    xmax = 2.0                  # physical domain parameters
    uf = np.full(nx, 0.2)       # velocity at faces (assume constant)
    
    keep_model_stable = False
    if keep_model_stable == True:
        cmax = 1.
        dxcmin = np.min(0.5*dt*(np.roll(uf,-1) + uf)/cmax)
    else:
        dxcmin = 0.
        
    xf, dxc, xc, dxf = gr.coords_centralstretching(xmax, nx, nx/2, dxcmin=dxcmin) # points in space, length of spatial step
    #xf, dxc, xc, dxf = gr.coords_uniform(xmax, nx) # points in space, length of spatial step
    print(nx/2)
    uc = gr.linear(xc, xf, uf)       # velocity at centers
    cc = 0.5*dt*(np.roll(uf,-1) + uf)/dxc # Courant number (defined at cell center)
    niter = 1                   # number of iterations (for Jacobi or Gauss-Seidel)
    
    #ut.make_animation('Upwind', 'Upwind_nt100', nt, dt, uf, dxc, xc, xmax, uc)

    # Print and plot grid and Courant number
    print('The (cell center) points and Courant numbers are:')
    for i in range(nx):
        print(i, "%.2f" %xc[i], "%.2f" %cc[i])
    print()
    ut.plot_Courant(xc, cc)
    ut.plot_grid(xc, dxc)

    # Calculate initial functions
    psi1_in = an.analytic1(xc, xmax)
    psi2_in = an.analytic2(xc, xmax)

    # Calculate analytic solutions
    psi1_an = an.analytic1(xc, xmax, uc, nt*dt)
    psi2_an = an.analytic2(xc, xmax, uc, nt*dt)

    #################
    #### Schemes ####
    #################

    do_basicschemes = False
    basicschemes = []
    advancedschemes = ['BTBS_Jacobi', 'hybrid_Upwind_BTBS1J']#['BTBS_Jacobi', 'hybrid']
    markers_as = ['x', 'x', '+', '', '', '']
    linestyle_as = ['-','-','-', '--', '-', '--']
    colors_as = ['red', 'blue', 'orange', 'red', 'lightblue', 'gray']
    allschemes = basicschemes + advancedschemes

    # Calculate numerical results
    for s in allschemes:
        fn = getattr(sch, f'{s}')
        if 'Jacobi' in s or 'GaussSeidel' in s:
            locals()[f'psi1_{s}'] = fn(psi1_in.copy(), nt, dt, uf, dxc, niter)
            locals()[f'psi2_{s}'] = fn(psi2_in.copy(), nt, dt, uf, dxc, niter)
        else:
            locals()[f'psi1_{s}'] = fn(psi1_in.copy(), nt, dt, uf, dxc)
            locals()[f'psi2_{s}'] = fn(psi2_in.copy(), nt, dt, uf, dxc)
    
    ##########################
    #### Plotting schemes ####
    ##########################
    
    if do_basicschemes == True:
        plt.plot(xc, psi1_in, label='Initial', linestyle='-', color='grey')
        plt.plot(xc, psi1_an, label='Analytic', linestyle='-', color='k', marker='x')
        for s in basicschemes:
            plt.plot(xc, locals()[f'psi1_{s}'], label=f'{s}')
        ut.design_figure('Psi1_bs.pdf', f'$\\Psi_1$ at t={nt*dt} - Basic Schemes', \
                        'x', '$\\Psi_1$', True, -0.1, 1.5)


    plt.plot(xc, psi1_in, label='Initial', linestyle='-', color='grey')
    plt.plot(xc, psi1_an, label='Analytic', linestyle='-', color='k')
    for s in advancedschemes:
        si = advancedschemes.index(s)
        if 'Jacobi' in s or 'GaussSeidel' in s:
            slabel = f'{s}, it={niter}'
        elif s == 'BTBS':
            slabel = 'BTBS_numpy'
        else: 
            slabel = s
        plt.plot(xc, locals()[f'psi1_{s}'], label=f'{slabel}', marker=markers_as[si], linestyle=linestyle_as[si], color=colors_as[si])
    ut.design_figure('Psi1_as.pdf', f'$\\Psi_1$ at t={nt*dt}', \
                     'x', '$\\Psi_1$', True, -0.1, 1.5)

    if do_basicschemes == True:
        plt.plot(xc, psi2_in, label='Initial', linestyle='-', color='grey')
        plt.plot(xc, psi2_an, label='Analytic', linestyle='-', color='k', marker='x')
        for s in basicschemes:
            plt.plot(xc, locals()[f'psi2_{s}'], label=f'{s}')
        ut.design_figure('Psi2_bs.pdf', f'$\\Psi_2$ at t={nt*dt} - Basic Schemes', \
                        'x', '$\\Psi_2$', True, -0.1, 1.5)
    
    
    plt.plot(xc, psi2_in, label='Initial', linestyle='-', color='grey')
    plt.plot(xc, psi2_an, label='Analytic', linestyle='-', color='k')
    for s in advancedschemes:
        si = advancedschemes.index(s)
        if 'Jacobi' in s or 'GaussSeidel' in s:
            slabel = f'{s}, it={niter}'
        else: 
            slabel = s
        plt.plot(xc, locals()[f'psi2_{s}'], label=f'{slabel}', marker=markers_as[si], linestyle=linestyle_as[si], color=colors_as[si])
    ut.design_figure('Psi2_as.pdf', f'$\\Psi_2$ at t={nt*dt}', \
                     'x', '$\\Psi_2$', True,  -0.1, 1.5)
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
    csv_psi1_analytic = epm.check_conservation(psi1_in, psi1_an, dxc)
    print(f'1 - Total mass gained at t={nt*dt} - Analytic {csv_psi1_analytic:.2E}')    
    for s in allschemes:
        locals()[f'csv_psi1_{s}'] = epm.check_conservation(psi1_in, locals()[f'psi1_{s}'], dxc)
        print(f'1 - Total mass gained at t={nt*dt} - {s} {locals()[f'csv_psi1_{s}']:.2E}')

    csv_psi2_analytic = epm.check_conservation(psi2_in, psi2_an, dxc)
    print(f'2 - Total mass gained at t={nt*dt} - Analytic {csv_psi2_analytic:.2E}')   
    for s in allschemes:
        locals()[f'csv_psi2_{s}'] = epm.check_conservation(psi2_in, locals()[f'psi2_{s}'], dxc)
        print(f'2 - Total mass gained at t={nt*dt} - {s} {locals()[f'csv_psi2_{s}']:.2E}')
    
    """
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
    ut.design_figure(f'loglog_{scheme}.pdf', f'RMSE for {scheme} scheme', 'dx', 'RMSE')
    """

    print()

    #### Boundedness
    bdn_psi1_analytic = epm.check_boundedness(psi1_in, psi1_an)
    print(f'1 - Boundedness at t={nt*dt} - Analytic: {bdn_psi1_analytic}')    
    for s in allschemes:
        locals()[f'bdn_psi1_{s}'] = epm.check_boundedness(psi1_in, locals()[f'psi1_{s}'])
        print(f'1 - Boundedness at t={nt*dt} - {s}: {locals()[f'bdn_psi1_{s}']}')

    bdn_psi2_analytic = epm.check_boundedness(psi2_in, psi2_an)
    print(f'2 - Boundedness at t={nt*dt} - Analytic: {bdn_psi2_analytic}')   
    for s in allschemes:
        locals()[f'bdn_psi2_{s}'] = epm.check_boundedness(psi2_in, locals()[f'psi2_{s}'])
        print(f'2 - Boundedness at t={nt*dt} - {s}: {locals()[f'bdn_psi2_{s}']}')

if __name__ == "__main__": main()