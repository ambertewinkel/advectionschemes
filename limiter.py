# This file defines the limiter used in schemes.py functions.
# Author: Amber te Winkel
# Email: a.j.tewinkel@pgr.reading.ac.uk

import numpy as np
from numba_config import jitflags
from numba import njit
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def FCT(field_LO, corr, dxc, previous, double=False, secondfield=None):
    """
    This function limits the high-order correction based 
    on the low-order solution's local bounds. If previous=True, 
    then also the previous time step field can determine the max/min bounds.
    --- Input --- 
    field_LO: low-order solution
    corr: high-order correction, corr[i] is defined at i-1/2, A in Zalesak 1979
    dxc: cell width
    previous: previous time step field - if an element in previous is None, it is not used.
    double: double FCT limiter, True if FCT is applied twice, see doubleFCT function below.
    secondfield: additional field (e.g. initial low-order solution) to determine the max/min values, only used if double==True.
    --- Output ---
    corrlim: limited high-order correction, C*A in Zalesak 1979
    """
    n = len(field_LO)
    corrlim, C, fieldmax, fieldmin, Pp, Qp, Rp, Pm, Qm, Rm = np.zeros(n), np.zeros(n), np.zeros(n),  np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    for i in range(n):
        if corr[i]*(field_LO[i] - field_LO[i-1]) <= 0. and (corr[i]*(field_LO[(i+1)%n] - field_LO[i]) <= 0. or corr[i]*(field_LO[i-1] - field_LO[i-2]) <= 0.):
            corr[i] = 0.

        # Determine local max and min
        if previous[i] is not None and double == False:
            fieldmax[i] = max([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], previous[i-1], previous[i], previous[(i+1)%n]])
            fieldmin[i] = min([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], previous[i-1], previous[i], previous[(i+1)%n]])
        elif previous[i] is not None and double == True:
            fieldmax[i] = max([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], previous[i-1], previous[i], previous[(i+1)%n], secondfield[i-1], secondfield[i], secondfield[(i+1)%n]])
            fieldmin[i] = min([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], previous[i-1], previous[i], previous[(i+1)%n], secondfield[i-1], secondfield[i], secondfield[(i+1)%n]])
        elif previous[i] is None and double == True:
            fieldmax[i] = max([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], secondfield[i-1], secondfield[i], secondfield[(i+1)%n]])
            fieldmin[i] = min([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], secondfield[i-1], secondfield[i], secondfield[(i+1)%n]])
        else:
            fieldmax[i] = max([field_LO[i-1], field_LO[i], field_LO[(i+1)%n]])
            fieldmin[i] = min([field_LO[i-1], field_LO[i], field_LO[(i+1)%n]])

        Pp[i] = max([0., corr[i]]) - min([0., corr[(i+1)%n]])
        Qp[i] = (fieldmax[i] - field_LO[i])*dxc[i]
        Rp[i] = min([1., Qp[i]/Pp[i]]) if Pp[i] > 0. else 0.
        
        Pm[i] = max([0., corr[(i+1)%n]]) - min([0., corr[i]])
        Qm[i] = (field_LO[i] - fieldmin[i])*dxc[i]
        Rm[i] = min([1., Qm[i]/Pm[i]]) if Pm[i] > 0. else 0.

    for i in range(n):
        # Determine C at face i-1/2
        C[i] = min([Rp[i-1], Rm[i]]) if corr[i] < 0. else min([Rp[i], Rm[i-1]])

        # Determine limited correction
        corrlim[i] = C[i]*corr[i]

    return corrlim


def nonneg(field, flx, dxc): # assumes constant dxc
    """
    This function implements nonnegativity as discussed with HW in the 27-01-2025 meeting.
    field[i] defined at i, resulting field from the fluxes defined by flx
    flx[i] defined at i-1/2, the flux that leads to field being as it is
    """
    for i in range(len(field)):
        while field[i] < 0. and flx[i] < 0.:
            addflx = -field[i]*dxc[i] # field[i] = negative, addflx = positive
            flx[i] += addflx 
            field[i] += addflx/dxc[i] # adjust field[i]
            field[i-1] -= addflx/dxc[i] # adjust field[i-1] # assumes constant dxc
            i = (i - 1)%len(field) # adjust field[i-1] and check in the same loop whether it has become negative, if so limit flx[i-2] and so on until no negative values are found
        while field[i] < 0. and flx[(i + 1)%len(field)] > 0.:
            rmvflx = field[i]*dxc[i] # field[i] = negative, rmvflx = negative
            flx[(i + 1)%len(field)] += rmvflx 
            field[i] -= rmvflx/dxc[i] # adjust field[i]
            field[(i+1)%len(field)] += rmvflx/dxc[i] # adjust field[i+1] # assumes constant dxc
            i = (i + 1)%len(field) # adjust field[i+1] and check in the same loop whether it has become negative, if so limit flx[i+2] and so on until no negative values are found       

    return field


def doubleFCT_updateminmax(field_LO, corr, dxc, previous):
    """
    This function implements double FCT as discussed with HW in the 27-01-2025 meeting. Note: this does update the min/max in the second FCT application inside of the FCT function. This is different from the multiFCT below.
    """
    corrlim = FCT(field_LO, corr, dxc, previous) # first FCT application
    fieldlim = field_LO - (np.roll(corrlim,-1) - corrlim)/dxc # HO bounded solution after one FCT application
    newcorr = corr - corrlim
    newcorrlim = FCT(fieldlim, newcorr, dxc, previous, double=True, secondfield=field_LO) # second FCT application
    
    return corrlim + newcorrlim


def FCTnonneg(field_LO, corr, dxc, previous):
    """
    This function limits the high-order correction based 
    set bounds such that the field does not go negative. The lower bound is 0.0 and the upper bound is 1000000
    --- Input --- 
    field_LO: low-order solution
    corr: high-order correction, corr[i] is defined at i-1/2, A in Zalesak 1979
    dxc: cell width
    --- Output ---
    corrlim: limited high-order correction, C*A in Zalesak 1979
    """
    n = len(field_LO)
    corrlim, C, Pp, Qp, Rp, Pm, Qm, Rm = np.zeros(n),  np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    fieldmin, fieldmax = 0.0, 1000000.0

    for i in range(n):
        if corr[i]*(field_LO[i] - field_LO[i-1]) <= 0. and (corr[i]*(field_LO[(i+1)%n] - field_LO[i]) <= 0. or corr[i]*(field_LO[i-1] - field_LO[i-2]) <= 0.):
            corr[i] = 0.

        Pp[i] = max([0., corr[i]]) - min([0., corr[(i+1)%n]])
        Qp[i] = (fieldmax - field_LO[i])*dxc[i]
        Rp[i] = min([1., Qp[i]/Pp[i]]) if Pp[i] > 0. else 0.
        
        Pm[i] = max([0., corr[(i+1)%n]]) - min([0., corr[i]])
        Qm[i] = (field_LO[i] - fieldmin)*dxc[i]
        Rm[i] = min([1., Qm[i]/Pm[i]]) if Pm[i] > 0. else 0.

    for i in range(n):
        # Determine C at face i-1/2
        C[i] = min([Rp[i-1], Rm[i]]) if corr[i] < 0. else min([Rp[i], Rm[i-1]])

        # Determine limited correction
        corrlim[i] = C[i]*corr[i]

    return corrlim


#def doubleFCT_noupdate(field_LO, corr, dxc, previous, nFCT=1):
#    corrlim = FCT(field_LO, corr, dxc, previous) # first FCT application
#    fieldlim = field_LO - (np.roll(corrlim,-1) - corrlim)/dxc # HO bounded solution after one FCT application
#    newcorr = corr - corrlim
#    newcorrlim = FCT(fieldlim, newcorr, dxc, previous) # second FCT application
#
#    return corrlim + newcorrlim
#
#
#def multiFCT_noupdate(field_LO, corr, dxc, previous, nFCT=1):
#    totalcorrlim = np.zeros(len(corr))
#    newcorr = corr.copy()
#    boundedfield = field_LO.copy()
#    for f in range(nFCT):
#        corrlim = FCT(boundedfield, newcorr, dxc, previous)
#        boundedfield = field_LO - (np.roll(corrlim,-1) - corrlim)/dxc # HO bounded solution after one FCT application
#        newcorr = corr - corrlim
#        totalcorrlim += corrlim
#    
#    return totalcorrlim

def multiFCT(fieldlow, flxlow, fieldit, corr, dxc, previous, nFCT=1):
    """
    This function limits the high-order correction based 
    on the low-order solution's local bounds. If previous=True, 
    then also the previous time step field can determine the max/min bounds.
    --- Input --- 
    fieldlow: low-order solution
    corr: high-order correction, corr[i] is defined at i-1/2, A in Zalesak 1979
    dxc: cell width
    previous: previous time step field - if an element in previous is None, it is not used.
    double: double FCT limiter, True if FCT is applied twice, see doubleFCT function below.
    secondfield: additional field (e.g. initial low-order solution) to determine the max/min values, only used if double==True.
    --- Output ---
    corrlim: limited high-order correction, C*A in Zalesak 1979


    This function adapts the FCT function above to allow for multiple iterations of FCT, which update the low-order solution and its fluxes, but reuses the set max/min from the first iteration to ensure monotonicity.
    fieldit = field at previous timestep
    """
    field_LO = fieldlow.copy()
    flx_LO = flxlow.copy()
    corrcp = corr.copy()

    #plt.plot('fieldit', fieldit)
    #plt.plot('field_LO', field_LO)
    #plt.plot(corr/0.03125, label='corr', color='k')
    #plt.plot(flx_LO, label='flx_LO')
    #plt.legend()
    #plt.show()

    n = len(field_LO)
    corrlim, C, fieldmax, fieldmin, Pp, Qp, Rp, Pm, Qm, Rm = np.zeros(n), np.zeros(n), np.zeros(n),  np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    #field_LOtest = fieldit - (np.roll(flx_LO,-1) - flx_LO)/dxc
    #print('Test field_LO from flx_LO:', field_LOtest)
    #print('field_LO:', field_LO)
    for i in range(n):
        # Determine local max and min
        if previous[i] is not None:
            fieldmax[i] = max([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], previous[i-1], previous[i], previous[(i+1)%n]])
            fieldmin[i] = min([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], previous[i-1], previous[i], previous[(i+1)%n]])
        else:
            fieldmax[i] = max([field_LO[i-1], field_LO[i], field_LO[(i+1)%n]])
            fieldmin[i] = min([field_LO[i-1], field_LO[i], field_LO[(i+1)%n]])
        #!if corrcp[i]*(field_LO[i] - field_LO[i-1]) <= 0. and (corrcp[i]*(field_LO[(i+1)%n] - field_LO[i]) <= 0. or corrcp[i]*(field_LO[i-1] - field_LO[i-2]) <= 0.):
        #!        corrcp[i] = 0.
        #        #print(f'f={f}, i={i}, corrcp[i] = 0 from if statement') #!!! quite important! but not why full corrcp zero at even iterations. 

    #print('min', fieldmin)
    #print('max', fieldmax)
    # doesn't solve the min/max Qm Qp problem:
    # #fieldmax = np.maximum(np.roll(field_LO,1), np.maximum(field_LO, np.roll(field_LO,-1)))
    #fieldmin = np.minimum(np.roll(field_LO,1), np.minimum(field_LO, np.roll(field_LO,-1)))

    for f in range(nFCT):
        #print(f)
        for i in range(n):
            #!if corrcp[i]*(field_LO[i] - field_LO[i-1]) <= 0. and (corrcp[i]*(field_LO[(i+1)%n] - field_LO[i]) <= 0. or corrcp[i]*(field_LO[i-1] - field_LO[i-2]) <= 0.):
            #!    corrcp[i] = 0.
                #print(f'f={f}, i={i}, corrcp[i] = 0 from if statement') #!!! quite important! but not why full corrcp zero at even iterations. 

            Pp[i] = max([0., corrcp[i]]) - min([0., corrcp[(i+1)%n]])
            #Pp[i] = max([0., corrcp[i-1]]) - min([0., corrcp[i]])
            Qp[i] = (fieldmax[i] - field_LO[i])*dxc[i]
            if abs(Qp[i]) < 1e-12:
                Qp[i] = 0.
            #if Qp[i] < 0.:
            #    if Qp[i] > -1e-12: #!!! bad avoiding of problems??? - doesn't solve it.
            #        Qp[i] = 0.
            #Rp[i] = min([1., Qp[i]/Pp[i]]) if Pp[i] > 0. else 0.
            Rp[i] = min([1., Qp[i]/Pp[i]]) if Pp[i] > 1e-12 else 0.
            
            Pm[i] = max([0., corrcp[(i+1)%n]]) - min([0., corrcp[i]])
            #Pm[i] = max([0., corrcp[i]]) - min([0., corrcp[i-1]])
            Qm[i] = (field_LO[i] - fieldmin[i])*dxc[i]
            if abs(Qm[i]) < 1e-12:
                Qm[i] = 0.
            #if Qm[i] < 0.:
            #    if Qm[i] > -1e-12: #!!! bad avoiding of problems??? - doesn't solve it.
            #        Qm[i] = 0.
            #Rm[i] = min([1., Qm[i]/Pm[i]]) if Pm[i] > 0. else 0.
            Rm[i] = min([1., Qm[i]/Pm[i]]) if Pm[i] > 1e-12 else 0.
        
        #if f == 1:
        #    #print('Qp', Qp/dxc)
        #    #print('Qm', Qm/dxc)
        #    d=21
        #    #print('Qp', Qp)
        #    #print('Qm', Qm)
        #    print('it = ', f, 'i=', d)
        #    print('phiMax', fieldmax[d])
        #    print('phid', field_LO[d])
        #    print('Qp', Qp[d]/dxc[d])

        for i in range(n):
            # Determine C at face i-1/2
            C[i] = min([Rp[i-1], Rm[i]]) if corrcp[i] < 0. else min([Rp[i], Rm[i-1]])# !!! how can C possibly go negative?????
            #print(C[i])
            ####if C[i] < 0.:
            ####    print('iter', f+1, ': C[i] =', C[i], '< 0')
            ####    print('Rp[i-1]', Rp[i-1])
            ####    print('--> Qp[i-1]', Qp[i-1])
            ####    print('--> Pp[i-1]', Pp[i-1])
            ####    print('Rm[i]', Rm[i])
            ####    print('--> Qm[i]', Qm[i])
            ####    print('--> Pm[i]', Pm[i])
            ####    print('Rp[i]', Rp[i])
            ####    print('Rm[i-1]', Rm[i-1])
            ####    print('corrcp[i]', corrcp[i])
            ####    print()
            ####    #break
            # Determine limited correction 
            corrlim[i] += C[i]*corrcp[i] #!!! perhaps shift by 1? (i.e C[i+1])
        
        #print('corr', corr)
        #print('corrlim', corrlim)
        #print('corrcp', corrcp)
        # Recalculate field_LO
        #field_LO_old = field_LO.copy()
        #flx_LO_old = flx_LO.copy()
        flx_LO = flx_LO + corrlim # new bounded/monotonic flux # [i] defined at i-1/2
        field_LO = fieldit - (np.roll(flx_LO,-1) - flx_LO)/dxc # [i] defined at i
        #if f == 1:
        #    print(field_LO)
        #    break

        #plt.plot(fieldmax - field_LO_old, label=f'diff old LO iter={f+1}')
        #plt.plot(fieldmax - field_LO, label=f'diff new LO iter={f+1}')
        #plt.plot(corrlim/0.03125, label=f'corrlim iter={f+1}')
        # Update corrcp, the high-order correction (as there is a new bounded/monotonic field_LO)
        corrcp = corr - corrlim #corrcp - corrlim

        #plt.plot(Rp, label=f'Rp iter={f+1}')
        #plt.plot(Rm, label=f'Rm iter={f+1}')
        #plt.plot(C, label=f'C iter={f+1}')
        #plt.plot(corr, label=f'corr iter={f+1}')
        #plt.plot(corrcp, label=f'corrcp iter={f+1}')
        ##if f == 1:
        ##    print('iter', f+1, 'CA', corrlim/0.03125) 
            ##plt.plot(np.roll(C,-1), label=f'C iter={f+1}')#, marker='x')
            #plt.plot(Rp, label=f'Rp iter={f+1}')
            #plt.plot(Pp, label=f'Pp iter={f+1}')
            #plt.plot(Qp, label=f'Qp iter={f+1}')
            #plt.plot(Rm, label=f'Rm iter={f+1}')
            ##plt.plot(corrcp/0.03125, label=f'A iter={f+1}')
            ##plt.plot(C*corrcp/0.03125, label=f'C*A iter={f+1}')
        #plt.plot(corrlim/0.03125, label=f'corrlim iter={f+1}')
    #plt.ylim(-1,1)
    plt.legend()
    plt.axhline(0, color='k', linestyle='--')
    #plt.savefig('PPM_mFCT9_corrlim.pdf')
   # plt.show()

    return corrlim #corrcp#corrlim


############## HW CODE ##############


def advect(phi, c, flux, options=None):
    """Advect cell values phi using Courant number c using flux"""
    F = flux
    if callable(flux):
        F = flux(phi, c, options=options)
    return phi - c*(F - np.roll(F,1))
    
    
def PPMflux(phi, c,options=None):
    """Returns the PPM fluxes for cell values phi for Courant number c.
    Face j is at j+1/2 between cells j and j+1"""
    # Integer and remainder parts of the Courant number
    #print(phi)
    cI = int(c)
    cR = c - cI
    # phi interpolated onto faces
    phiI = 1/12*(-np.roll(phi,1) + 7*phi + 7*np.roll(phi,-1) - np.roll(phi,-2))
    # Move face interpolants to the departure faces
    if cI > 0:
        phiI = np.roll(phiI,cI)
    # Interface fluxes
    F = np.zeros_like(phi)
    # Contribution to the fluxes from full cells between the face and the departure point
    nx = len(F)
    for j in range(nx):
        for i in range(j-cI+1,j+1):
            F[j] += phi[i%nx]/c
        #F[j] += sum(phi[j-cI+1:j+1])
    # Ratio of remnamt to total Courant number
    cS = 1
    if c > 1:
        cS = cR/c
    # Contribution from the departure cell
    #print('First part of flux', F)
    
    #print('Second part of flux', cS*( (1 - 2*cR + cR**2)*phiI \
    #         + (3*cR - 2*cR**2)*np.roll(phi,cI) \
    #         + (-cR + cR**2)*np.roll(phiI,1)))
    #plt.plot(F, label='First HW')
    #plt.plot(cS*( (1 - 2*cR + cR**2)*phiI \
    #            + (3*cR - 2*cR**2)*np.roll(phi,cI) \
    #            + (-cR + cR**2)*np.roll(phiI,1)), label='Second HW')
    #plt.legend()
    #plt.show()
    
    F += cS*( (1 - 2*cR + cR**2)*phiI \
             + (3*cR - 2*cR**2)*np.roll(phi,cI) \
             + (-cR + cR**2)*np.roll(phiI,1))
    return F


def findMinMax(phid, phi, minPhi, maxPhi):
    """Return phiMin and phiMax for bounded solutions. 
    If minPhi and maxPhi are not none, these are the values
    If phi is not None, find nearest neighbours of phid and phi to determin phiMin and phiMax
    Suitable for c<=1
    If phi is None, just use phid. Suitable for all c but more diffusive."""
    phiMax = maxPhi
    if phiMax is None:
        phiMax = phid
        if phi is not None:
            phiMax = np.maximum(phi, phiMax)
        phiMax = np.maximum(np.roll(phiMax,1), np.maximum(phiMax, np.roll(phiMax,-1)))
    
    phiMin = minPhi
    if phiMin is None:
        phiMin = phid
        if phi is not None:
            phiMin = np.minimum(phi, phiMin)
        phiMin = np.minimum(np.roll(phiMin,1), np.minimum(phiMin, np.roll(phiMin,-1)))

    return phiMin, phiMax


def upwindMatrix(c, a, nx):
    # Matrix for implicit solution of
    # phi_j^{n+1} = phi_j^n - (1-a)*c(phi_j^n - phi_{j-1}^n)
    #                       - a*c*(phi_j^{n+1} - phi_{j-1}^{n+1}
    return diags([-a*c*np.ones(nx-1),  # The diagonal for j-1
                 (1+a*c)*np.ones(nx), # The diagonal for j
                 [-a*c]], # The top right corner for j-1
                 [-1,0,nx-1], # the locations of each of the diagonals
                 shape=(nx,nx), format = 'csr')


def alpha(c):
    "Off-centering for implicit solution"
    return 1-1/np.maximum(c, 1)


def upwindFlux(phi, c, options=None):
    """Returns the first-order upwind fluxes for cell values phi and Courant number c
    Implicit or explicit depending on Courant number"""
    if not isinstance(options, dict):
        options = {}
    explicit =  options["explicit"] if "explicit" in options else (c <= 1)
    if explicit:
        return phi
    nx = len(phi)
    # Off centering for Implicit-Explicit
    a = alpha(c)
    M = upwindMatrix(c, a, nx)
    # Solve the implicit problem
    phiNew = spsolve(M, phi - (1-a)*c*(phi - np.roll(phi,1)))
    # Back-substitute to get the implicit fluxes
    return (1-a)*phi + a*phiNew


def FCT_HW(phi, c, options={"HO":PPMflux, "LO":upwindFlux, "nCorr":1, 
                         "minPhi": None, "maxPhi": None}):
    """Returns the corrected high-order fluxes with nCorr corrections"""
    # Sort out options
    if not isinstance(options, dict):
        options = {}
    HO =  options["HO"] if "HO" in options else PPMflux
    LO =  options["LO"] if "LO" in options else upwindFlux
    nCorr = options["nCorr"] if "nCorr" in options else 1
    minPhi = options["minPhi"] if "minPhi" in options else None
    maxPhi = options["maxPhi"] if "maxPhi" in options else None
    
    # First approximation of the bounded flux and the full HO flux
    fluxB = LO(phi,c, options=options)
    fluxH = HO(phi,c, options=options)

    # The first bounded solution
    phid = advect(phi, c, fluxB)

    # The allowable min and max
    if c <= 1:
        phiMin, phiMax = findMinMax(phid, phi, minPhi, maxPhi)
    else:
        phiMin, phiMax = findMinMax(phid, None, minPhi, maxPhi)

    #print('phiMin', phiMin)
    #print('phiMax', phiMax)

    #plt.plot(fluxH - fluxB, label='corr', color='k')
    #plt.plot(fluxB, label='flx_LO')
    #plt.plot(fluxH, label='flx_HO')
    #plt.legend()
    #plt.show()
    total = np.zeros(len(phi))
    # Add a corrected HO flux
    for it in range(nCorr):
        # The antidiffusive fluxes
        A = fluxH - fluxB

        # Sums of influxes ad outfluxes
        Pp = c*(np.maximum(0, np.roll(A,1)) - np.minimum(0, A))
        Pm = c*(np.maximum(0, A) - np.minimum(0, np.roll(A,1)))

        # The allowable rise and fall using an updated bounded solution
        if it > 0:
            phid = advect(phi, c, fluxB)
            #if it == 2:
                #print(phid)
            #    break
        Qp = phiMax - phid
        Qm = phid - phiMin
        if it == 1:
            d=21
            #print('Qp', Qp)
            #print('Qm', Qm)
            print('it = ', it, 'i=', d)
            print('phiMax', phiMax[d])
            print('phid', phid[d])
            print('Qp', Qp[d])

        # Ratios of allowable to HO fluxes
        Rp = np.where(Pp > 1e-12, np.minimum(1, Qp/np.maximum(Pp,1e-12)), 0)
        Rm = np.where(Pm > 1e-12, np.minimum(1, Qm/np.maximum(Pm,1e-12)), 0)

        # The flux limiter
        C = np.where(A >= 0, np.minimum(np.roll(Rp,-1), Rm),
                             np.minimum(Rp, np.roll(Rm,-1)))
        fluxB = fluxB + C*A

        total += C*A
        ##if it == 1:
            #plt.plot(C, label='C iter '+str(it+1))
            ##plt.plot(Rp, label='Rp iter '+str(it+1))
            ##plt.plot(Rm, label='Rm iter '+str(it+1))
            #plt.plot(A, label='A iter '+str(it+1))
            #plt.plot(C*A, label='CA iter '+str(it+1))
        ##    print('iter', it+1, 'CA', C*A)
        #plt.plot(total, label='Total after iteration '+str(it+1))
    #plt.legend()
    #plt.show()
        
    return fluxB