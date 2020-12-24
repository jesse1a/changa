import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cython
import pytest
import scipy
import astropy
import astropy.units as u
import gala.dynamics as gd
from gala.units import UnitSystem
from gala.units import galactic
import gala.integrate as gi
import superfreq
from superfreq.core import SuperFreqResult
from collections import Counter
from timeit import default_timer as timer

import pynbody

nSteps = 1100 # Number of time-steps to look at; should be same number as nSteps in orbit_integration.param
nParticles = 10000 # Number of particles to follow

# Phase-space coordinates for each particle over time:
x = np.empty((nSteps,nParticles))
y = np.empty((nSteps,nParticles))
z = np.empty((nSteps,nParticles))
vx = np.empty((nSteps,nParticles))
vy = np.empty((nSteps,nParticles))
vz = np.empty((nSteps,nParticles))

iOutInterval = 1.0 # Should be same number as iOutInterval in orbit_integration.param; nSteps/iOutInterval = total number of .tipsy files generated
start = timer()
for i in range(nSteps): # Loop through each time step
    begin = timer()
    n = str(i+1) # Corresponds to iOutInterval = 1.0
    snapshot = pynbody.load('[insert full directory pathwway here]/orbit_integration.'+n.zfill(6)) # Snapshot of phase-space information for all particles implemented by GalactICS
    snapshot = snapshot[200000:210000] # Only look at particles 200,000 to 210,000 out of 1,000,000; number of integers in half-open interval [200000,210000) corresponds exactly to nParticles
    # Phase-space coordinates for each followed particle at time i:
    x[i] = snapshot['x']
    y[i] = snapshot['y']
    z[i] = snapshot['z']
    vx[i] = snapshot['vx']
    vy[i] = snapshot['vy']
    vz[i] = snapshot['vz']
    print('Time to set phase space coordinates for snapshot',i+1,'(in sec):',timer()-begin)
print('Total time to load snapshots & create phase space matrices (in min):',(timer()-start)/60)
# Make each row of position/velocity matrix correspond to time series for single particle:
x = np.transpose(x)
y = np.transpose(y)
z = np.transpose(z)
vx = np.transpose(vx)
vy = np.transpose(vy)
vz = np.transpose(vz)

dDelta = 0.5 # Corresponds to amount of time in a single timestep; should be same number as dDelta in orbit_integration.param
T = np.linspace(0,dDelta*nSteps,nSteps) # Array of times within total integration time period
freq_class = superfreq.SuperFreq(T,keep_calm=True) # Implements the Numerical Analysis of Fundamental Frequencies method for time period T, developed by Laskar and modified by Valluri and Merritt; keep_calm=True ensures that the fundamental frequency finder moves on to the next frequency component and does not throw a RuntimeError if it fails to determine a particular component
fy_ratio = np.zeros(nParticles) # Set of fundamental frequency ratios omega_y/omega_x for all followed particles
fz_ratio = np.zeros(nParticles) # Set of fundamental frequency ratios omega_z/omega_x for all followed particles
start = timer()
for k in range(nParticles): # Loop over all followed particles
    xfreqs,xamps,xphis = freq_class.frecoder(x[k] + 1j*vx[k]) # xfreqs = array of frequencies obtained from the Fourier transform of the time-series x[k] + i*vx[k] convolved with a Hanning filter; xamps = array of real amplitudes corresponding to the frequency components in xfreqs; xphis = array of phases corresponding to the frequency components in xfreqs
    yfreqs,yamps,yphis = freq_class.frecoder(y[k] + 1j*vy[k]) # Same as above, except for y-components
    zfreqs,zamps,zphis = freq_class.frecoder(z[k] + 1j*vz[k]) # Same as above, except for z-components
    if k == 0:
        print('Time elapsed (in sec) to obtain frequencies of first particle`s orbit =',timer()-start)
    # omega_x = frequency in xfreqs corresponding to the strongest amplitude; similar for omega_y and omega_z
    fy_ratio[k] = yfreqs[yamps.argmax()]/xfreqs[xamps.argmax()] # omega_y/omega_x for particle k
    fz_ratio[k] = zfreqs[zamps.argmax()]/xfreqs[xamps.argmax()] # omega_z/omega_x for particle k
    
    if np.remainder(k,1000) == 0:
        print('k =',k,'Time elapsed (in min) =',(timer()-start)/60)
    # Plot some orbits in unusual fundamental frequency ratio ranges to probe for irregularities:
    #if (0.5 < fy_ratio[k] < 0.8) or (1.2 < fy_ratio[k] < 1.5):
    #    print('This is a test for particle #',k)
    #    print('w_y/w_x, w_z/w_x =',fy_ratio[k],',',fz_ratio[k])
    #    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    #    ax1.plot(x[k],y[k])
    #    ax1.set_title('y vs. x')
    #    ax2.plot(x[k],z[k])
    #    ax2.set_title('z vs. x')
    #    ax3.plot(y[k],z[k])
    #    ax3.set_title('z vs. y')
    #    plt.show()
print('Total time to run frequency analysis (in min):',(timer()-start)/60)

# Plot frequency map, omega_z/omega_x vs. omega_y/omega_x:
plt.scatter(fy_ratio,fz_ratio,s=1)
plt.tight_layout()
plt.show()
