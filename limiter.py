# This file defines the limiter used in schemes.py functions.
# Author: Amber te Winkel
# Email: a.j.tewinkel@pgr.reading.ac.uk

import numpy as np
from numba_config import jitflags
from numba import njit
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import schemes as sch
import spatialdiscretisations as sd

def FCT(field_LO, corr, dxc, previous, double=False, secondfield=None):
    """
    This function limits the high-order correction based 
    on the low-order solution's local bounds. If previous=True, 
    then also the previous time step field can determine the max/min bounds.
    --- Input --- 
    field_LO: low-order solution
    corr: high-order correction, corr[i] is defined at i-1/2
    dxc: cell width
    previous: previous time step field - if an element in previous is None, it is not used.
    double: double FCT limiter, True if FCT is applied twice, see doubleFCT function below.
    fieldlim: limited high-order field after one FCT application, only used if double==True.
    --- Output ---
    corrlim: limited high-order correction
    """
    n = len(field_LO)
    corrlim, C, fieldmax, fieldmin, Pp, Qp, Rp, Pm, Qm, Rm = np.zeros(n), np.zeros(n), np.zeros(n),  np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    for i in range(n):
        if corr[i]*(field_LO[i] - field_LO[i-1]) <= 0. and (corr[i]*(field_LO[(i+1)%n] - field_LO[i]) <= 0. or corr[i]*(field_LO[i-1] - field_LO[i-2]) <= 0.):
            corr[i] = 0.

        # Determine local max and min
        if previous[i] is not None and double == False:
            fieldmax[i] = max([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], previous[i-1], previous[i], previous[(i+1)%n]]) # 01-07-2025: I think incorrect to use i-1 and i+1 from previous when those are independently set and could be None (aren't automatically the previous field). 
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

    #plt.plot(Rp, label='Rp FCT')
    #plt.plot(Rm, label='Rm FCT')
    #plt.legend()
    #plt.show()

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


def doubleFCT(field_LO, corr, dxc, previous):
    """
    This function implements double FCT as discussed with HW in the 27-01-2025 meeting.
    """
    corrlim = FCT(field_LO, corr, dxc, previous) # first FCT application
    fieldlim = field_LO - (np.roll(corrlim,-1) - corrlim)/dxc # HO bounded solution after one FCT application
    newcorr = corr - corrlim
    newcorrlim = FCT(fieldlim, newcorr, dxc, previous, double=True, secondfield=field_LO) # second FCT application
    
    return corrlim + newcorrlim


def advect(phi, c, flux, options=None): # Author: HW
    """Advect cell values phi using Courant number c using flux"""
    F = flux
    if callable(flux):
        F = flux(phi, c, options=options)
    return phi - c*(F - np.roll(F,1))


def findMinMax(phid, phi, minPhi, maxPhi): # Author: HW
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


def upwindMatrix(c, a, nx): # Author: HW
    # Matrix for implicit solution of
    # phi_j^{n+1} = phi_j^n - (1-a)*c(phi_j^n - phi_{j-1}^n)
    #                       - a*c*(phi_j^{n+1} - phi_{j-1}^{n+1}
    c = c[0] # !!! assumes uniform Courant number
    a = a[0] # !!! assumes uniform Courant number
    return diags([-a*c*np.ones(nx-1),  # The diagonal for j-1
                 (1+a*c)*np.ones(nx), # The diagonal for j
                 [-a*c]], # The top right corner for j-1
                 [-1,0,nx-1], # the locations of each of the diagonals
                 shape=(nx,nx), format = 'csr')


def alpha(c): # Author: HW
    "Off-centering for implicit solution"
    return 1-1/np.maximum(c, 1)


def upwindFlux(phi, c, options=None): # Author: HW
    """Returns the first-order upwind fluxes for cell values phi and Courant number c
    Implicit or explicit depending on Courant number"""
    if not isinstance(options, dict):
        options = {}
    explicit =  options["explicit"] if "explicit" in options else (c[0] <= 1) # assumes uniform Courant number
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


def MULES(field, flx_HO, c, flx_b=upwindFlux, nIter=1, minField=None, maxField=None):
    """This function implements the MULES limiter as in 31-01-2025 MULES vs FCT pdf. It allows for multiple iterations of MULES.
    Input: field = previous timesteps field
    Output:
    """
    if flx_b is upwindFlux: # else: it is already just a flux value
        flx_b = flx_b(field, c) # sets flx_b at i+1/2
        # Calculate the low-order bounded solution
        field_b = advect(field, c, flx_b) # this function assumes flx_b is at i+1/2
        flx_b = np.roll(flx_b,1) # flx_b[i] is now at i-1/2
    else: 
        if callable(flx_b):
            flx_b = flx_b(field, c) # flx_b[i] is here at i-1/2
        # Calculate the low-order bounded solution
        field_b = advect(field, c, np.roll(flx_b,-1)) # this function assumes flx_b is at i+1/2

    # The allowable min and max
    if c[0] <= 1: # assumes uniform Courant number
        minval, maxval = findMinMax(field_b, field, minField, maxField)
    else:
        minval, maxval = findMinMax(field_b, None, minField, maxField)

    # Calculate the flux correction
    corr = flx_HO - flx_b

    Qp = maxval - field_b
    Qm = field_b - minval
    Pp = c*(np.maximum(0, corr) - np.minimum(0, np.roll(corr,-1)))
    Pm = c*(np.maximum(0, np.roll(corr,-1)) - np.minimum(0, corr))
    
    l = np.ones(len(corr))
    for ni in range(nIter):
        Ppprime = c*(np.maximum(0., l*corr) - np.minimum(0., np.roll(l*corr,-1)))
        Pmprime = c*(np.maximum(0., np.roll(l*corr,-1)) - np.minimum(0., l*corr))
        Rp = np.where(Pp > 0., np.minimum(1., (Qp + Pmprime)/(Pp + 1e-12)), 0.)
        Rm = np.where(Pm > 0., np.minimum(1., (Qm + Ppprime)/(Pm + 1e-12)), 0.)
        l = np.minimum(l, np.where(corr >= 0., np.minimum(Rp, np.roll(Rm,1)), np.minimum(np.roll(Rp,1), Rm)))

    return flx_b + l*corr


def iterFCT(flx_HO, dxc, dt, uf, C, field_previous, previous=None, niter=1, ymin=None, ymax=None):
    """This function implements iterative FCT, as described in MULES_HW.pdf (/strang_carryover_1d paper as MULES_HW.pdf has some notational problems)
    ymax and ymin are overall global min/max values if set.
    
    previous: previous time step field - if an element in previous is None, it is not used. # 01-07-2025: previous is defined at face i-1/2
    02-07-2025: dxc assumed constant. uf and C assumed potentially nonconstant (and potentially negative??) -> for this, we need to make sure that the velocity is part of the flux, as it determines how much mass is going I/O!
    beta off-centering for AdImEx Upwind is assumed to be max(0,1-1/C) to ensure monotonicity.
    02-07-2025: flx_HO needs to be u*field at face. before just field at face, excluding u
    """
    nx = len(flx_HO)
    flx_bounded, field_bounded = np.zeros(nx), np.zeros(nx)

    # Calculate bounded field and bounded flux - 1 time step of AdImEx Upwind low-order bounded solution (this will subsequently be updated in FCT iteration loop)
    beta = np.maximum(0., 1. - 1./C) # [i] at i-1/2
    M = np.zeros((nx, nx))
    for i in range(nx):
        M[i,i] = 1. + beta[i]*C[i]
        M[i,(i-1)%nx] = -1.*beta[i]*C[i]
    c1mbfield = C*(1-beta)*field_previous # [i] at i-1/2
    rhs = field_previous - (c1mbfield - np.roll(c1mbfield,1))
    field_bounded = np.linalg.solve(M, rhs) # [i] at i
    #flx_bounded = (1. - beta)*np.roll(field_previous,1) + beta*np.roll(field_bounded,1) # [i] is at i-1/2 # (the field value at i-1/2 that gives you the flux when *uf) based on the AdImEx Upwind bounded field # Assumes uf and C is positive
    flx_bounded = 0.5*(uf + np.abs(uf))*((1. - beta)*np.roll(field_previous,1) + beta*np.roll(field_bounded,1)) + 0.5*(uf - np.abs(uf))*((1. - beta)*field_previous + beta*field_bounded) # [i] is at i-1/2 ##!!! not anymore (the field value at i-1/2 that gives you the flux when *uf) based on the AdImEx Upwind bounded field # Assumes uf and C is positive -- not anymore!

    # Set allowable min and max values (not iterated over!)
    fieldmin, fieldmax = set_extrema(nx, uf, field_bounded, field_previous, previous, ymin, ymax) 

    # FCT iteration loop
    for iiter in range(niter):
        # Calculate high-order correction
        corr = flx_HO - flx_bounded # [i] at i-1/2 # uf*field i.e. including uf!!

        # Checking for rare cases where we need to set corr to zero # We do really need this for monotonicity! Fixed bug at 30-06-2025 with dip in the center of the cosine bell field after FCT excluding this part.
        for i in range(nx):
            if corr[i]*(field_bounded[i] - field_bounded[i-1]) <= 0. and (corr[i]*(field_bounded[(i+1)%nx] - field_bounded[i]) <= 0. or corr[i]*(field_bounded[i-1] - field_bounded[i-2]) <= 0.):
                corr[i] = 0. # !!! Should this depend on the sign of uf? -> 02-07-2025: fixed

        # Calculate allowable mass I/O for max rise and fall
        Qp = dxc*(fieldmax - field_bounded) # [i] at i
        Qm = dxc*(field_bounded - fieldmin) # [i] at i

        # Calculate I/O fluxes at cell centers
        #face_flux = uf*corr # [i] at i-1/2
        #Pp = dt*(np.maximum(0, face_flux) - np.minimum(0, np.roll(face_flux,-1)))
        #Pm = dt*(np.maximum(0, np.roll(face_flux,-1)) - np.minimum(0, face_flux)) 
        #face_flux = uf*corr # [i] at i-1/2
        Pp = dt*(np.maximum(0, corr) - np.minimum(0, np.roll(corr,-1)))
        Pm = dt*(np.maximum(0, np.roll(corr,-1)) - np.minimum(0, corr))

        # Calculate ratios of allowable (Q) to existing high-order (P) fluxes
        Rp = np.where(Pp > 1e-12, np.minimum(1., Qp/np.maximum(Pp,1e-12)), 0.) 
        Rm = np.where(Pm > 1e-12, np.minimum(1., Qm/np.maximum(Pm,1e-12)), 0.)

        # Calculate the limiter for each face
        face_limiter = np.where(corr >= 0., np.minimum(Rp, np.roll(Rm,1)), np.minimum(np.roll(Rp,1), Rm)) # [i] at i-1/2
        
        # Update the bounded flux and field
        flx_bounded += face_limiter*corr # thus this includes uf as well
        #field_bounded = field_previous - dt/dxc*(np.roll(uf*flx_bounded, -1) - uf*flx_bounded)
        field_bounded = field_previous - dt/dxc*(np.roll(flx_bounded, -1) - flx_bounded)

    # Output limited field[it+1] = field_bounded after niter iterations
    return field_bounded


def set_extrema(nx, uf, field_bounded, field_previous, previous=None, only_global=False, ymin=None, ymax=None):
    """This function returns the min and max values allowed for each grid cell, defined at center. Used in iterFCT."""

    fieldmin, fieldmax = np.zeros(nx), np.zeros(nx)
    if not only_global:
        for i in range(nx):
            if previous[i]: # 01-07-2025: previous[i] is at i-1/2 so for cell i we check whether at least one of previous[i] or previous[i+1] is True (i.e. whether C is smaller than 1 on both sides) but what about u velocity? 
                if uf[i] > 0.: # upwind value can be taken for max # !!! discuss 2 or 3 values with HW
                    fieldmax[i] = max([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx], field_previous[i-1], field_previous[i]])#, field_previous[(i+1)%nx]])
                    fieldmin[i] = min([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx], field_previous[i-1], field_previous[i]])#, field_previous[(i+1)%nx]])
                elif uf[i] < 0.:
                    fieldmax[i] = max([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx], field_previous[i], field_previous[(i+1)%nx]])
                    fieldmin[i] = min([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx],  field_previous[i], field_previous[(i+1)%nx]])
                else: 
                    fieldmax[i] = max([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx], field_previous[i]])
                    fieldmin[i] = min([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx],  field_previous[i]])
            else:
                fieldmax[i] = max([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx]])
                fieldmin[i] = min([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx]])
    # For global allowable min/max
    if ymin is not None:
        fieldmin = np.where(fieldmin < ymin, ymin, fieldmin)
    if ymax is not None:
        fieldmax = np.where(fieldmax > ymax, ymax, fieldmax)

    return fieldmin, fieldmax
    

def FCT_HW(phi, c, fluxB, fluxH, nCorr=1):#options={"HO":PPMflux, "LO":upwindFlux, "nCorr":1, 
                         #"minPhi": None, "maxPhi": None}):
    """Returns the corrected high-order fluxes with nCorr corrections"""
    # Sort out options
    #if not isinstance(options, dict):
    #    options = {}
    #HO =  options["HO"] if "HO" in options else PPMflux
    #LO =  options["LO"] if "LO" in options else upwindFlux
    #nCorr = options["nCorr"] if "nCorr" in options else 1
    #minPhi = options["minPhi"] if "minPhi" in options else None
    #maxPhi = options["maxPhi"] if "maxPhi" in options else None
    
    # First approximation of the bounded flux and the full HO flux
    #fluxB = LO(phi,c, options=options)
    #fluxH = HO(phi,c, options=options)

    #plt.plot(np.roll(fluxB,-1), label='flx_LO HW')
    #plt.plot(np.roll(fluxH,-1), label='flx_HO HW')
    #plt.plot(phi, label='field_previous HW')
#
    #plt.legend()
    #plt.show()
    
    # The first bounded solution
    phid = advect(phi, c, fluxB)

    # The allowable min and max
    if c[0] <= 1: # assumes uniform Courant number
        phiMin, phiMax = findMinMax(phid, phi, minPhi=None, maxPhi=None)
    else:
        phiMin, phiMax = findMinMax(phid, None, minPhi=None, maxPhi=None)

    #plt.plot(phi, label='phi_in')
    #plt.plot(phid, label='phid')
    #plt.plot(np.roll(fluxB,-1), label='flx_LO HW before')

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

        #plt.plot(fluxH, label='flx_HO')
        #plt.plot(fluxB, label='flx_LO')
        #plt.plot(A, label='A')
        #plt.legend()
        #plt.show()
        # The allowable rise and fall using an updated bounded solution
        if it > 0:
            phid = advect(phi, c, fluxB)
            #if it == 2:
                #print(phid)
            #    break

        # Removing this does not remove the nonmonotonic problem
        #for i in range(len(A)): # Check for special cases in which to fully remove the high-order fluxes (see Zalesak 1979)
        #    #if A[i-1]*(phid[i] - phid[i-1]) <= 0. and (A[i-1]*(phid[(i+1)%len(A)] - phid[i]) <= 0. or A[i-1]*(phid[i-1] - phid[i-2]) <= 0.):
        #    #    print('this is applied', i)
        #    #    A[i-1] = 0.
        #    if A[i]*(phid[(i+1)%len(A)] - phid[i]) < 0. and (A[i]*(phid[(i+2)%len(A)] - phid[(i+1)%len(A)]) < 0. or A[i]*(phid[i] - phid[i-1]) < 0.):
        #        print('this is applied', i)
        #        A[i] = 0.

        # Sums of influxes ad outfluxes
        Pp = c*(np.maximum(0, np.roll(A,1)) - np.minimum(0, A))
        Pm = c*(np.maximum(0, A) - np.minimum(0, np.roll(A,1)))
        #dt = 0.01
        ##uf = 3.575
        #dxc = 0.025
        #Pp = uf*dt*(np.maximum(0, np.roll(A,1)) - np.minimum(0, A))
        #Pm = uf*dt*(np.maximum(0, A) - np.minimum(0, np.roll(A,1)))

        Qp = phiMax - phid
        Qm = phid - phiMin
        #Qp = (phiMax - phid)*dxc
        #Qm = (phid - phiMin)*dxc
        #if it == 1:
        #    d=21
        #    #print('Qp', Qp)
        #    #print('Qm', Qm)
        #    print('it = ', it, 'i=', d)
        #    print('phiMax', phiMax[d])
        #    print('phid', phid[d])
        #    print('Qp', Qp[d])


        # Ratios of allowable to HO fluxes
        Rp = np.where(Pp > 0., np.minimum(1, Qp/np.maximum(Pp,1e-12)), 0.)
        Rm = np.where(Pm > 0., np.minimum(1, Qm/np.maximum(Pm,1e-12)), 0.)
        print(Rp[28])

        plt.plot(Pp[27:30], label='Pp')
        plt.plot(Pm[27:30], label='Pm')
        plt.plot(Qp[27:30], label='Qp')
        plt.plot(Qm[27:30], label='Qm')        
        plt.plot(Rp[27:30]/10., label='Rp')
        plt.plot(Rm[27:30]/10., label='Rm')
        plt.axhline(0, color='k', linestyle='--')
        plt.legend()
        plt.show()

        # The flux limiter
        C = np.where(A >= 0, np.minimum(np.roll(Rp,-1), Rm),
                             np.minimum(Rp, np.roll(Rm,-1))) # defined at i+1/2
        fluxB = fluxB + C*A

        #plt.plot(Rp, label='Rp iter '+str(it+1))
        #plt.plot(Rm, label='Rm iter '+str(it+1))
        #plt.plot(C, label='limiter iter'+str(it+1))
        #plt.legend()
        #plt.show()
#
        #exit()

        total += C*A
        ##if it == 1:
            #plt.plot(C, label='C iter '+str(it+1))
            ##plt.plot(Rp, label='Rp iter '+str(it+1))
            ##plt.plot(Rm, label='Rm iter '+str(it+1))
            #plt.plot(A, label='A iter '+str(it+1))
            #plt.plot(C*A, label='CA iter '+str(it+1))
        ##    print('iter', it+1, 'CA', C*A)
        #plt.plot(total, label='Total after iteration '+str(it+1))

    print('C', C[26:30])
    print('A', A[26:30])
    print('C*A', (C*A)[26:30])    

    plt.plot(C[26:30], label='C')
    plt.plot(A[26:30], label='A')
    plt.plot((C*A)[26:30], label='C*A')
    plt.axhline(0, color='k', linestyle='--')
    plt.legend()
    plt.show()

    plt.plot(C, label='C')
    plt.plot(A, label='A')
    plt.plot((C*A), label='C*A')
    plt.axhline(0, color='k', linestyle='--')
    plt.legend()
    plt.show()
    
    #plt.legend()
    #plt.show()

    phiH = advect(phi, c, fluxH)
    phiFCT = advect(phi, c, fluxB)
    plt.plot(phiH, label='phiH')
    plt.plot(phiFCT, label='phiFCT')
    ##plt.plot(np.roll(fluxB,-1), label='flx_LO HW after')
    ##plt.plot(np.roll(fluxH,-1), label='flx_HO HW')
    plt.legend()
    #plt.show()

    fluxsum = C*A + np.roll(C*A,1)
    plt.plot(fluxsum, label='fluxsum')
    plt.axhline(0, color='k', linestyle='--')
    plt.legend()
    #plt.show()

    #print(phiFCT[26:31])
    print('Rp[28]', Rp[28])
    print('Rm[27]', Rm[27])
    print('Rm[27] supp', np.roll(Rm,-1)[28])   

    for i in range(len(A)):
        if Rm[i-1] != np.roll(Rm,1)[i]:
            print('Rm not equal at i-1/2', i, Rm[i-1], np.roll(Rm,1)[i])

    return fluxB