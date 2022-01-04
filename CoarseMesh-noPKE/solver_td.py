# Class to solve the time-dependent problem
# in an implicit scheme

import copy
import numpy as np
from scipy.sparse import coo_matrix


class SolveTD:

###############################################################
	
	def __init__(self, opts, flux, C, kcrit, T):

		N = opts.numBins*opts.numBins
		self.flux = flux
		self.Src  = np.zeros(N*opts.numGroups)
		self.DopplerFactor = np.zeros(N)
		self.kcrit = kcrit
		self.C = C
		self.T = T
		
###############################################################

	def gauss_seidel(self, opts, D, O, XS, start_flux, time, wd1, wd2, wp):

		# Initialize local variables
		dt = opts.dt
		N  = opts.numBins*opts.numBins
		PrecursorFactor = np.zeros(2)
		RRfis = np.zeros([N])
		RRfis_mod = np.zeros([2,N])
		mod1 = np.zeros(2)
		mod2 = np.zeros(2)
		dTdt  = np.zeros(N)
		KAPPA = 3.204e-11 # [J/fission]
		ALPHA = 3.83e-11  # [K cm3]
		GAMMA = 3.034e-3  # [K^[-1/2]]
		ERRflux = 1000.
		count = 0
		wd = [wd1, wd2]

		# Update S vector
		SumC = np.dot(XS.DECAYi*np.exp(-(XS.DECAYi[:]+wd[:])*opts.dt),self.C)
		fission_term = self.flux[:N]*XS.fis[:N]*sum(
			(XS.Bi[:]*XS.DECAYi[:])/(XS.DECAYi[:]+wp)/self.kcrit * 
			(-np.exp(-(XS.DECAYi[:]+wp)*opts.dt)+(1-np.exp(-(XS.DECAYi[:]+wp)*opts.dt))/((XS.DECAYi[:]+wp)*opts.dt)))		
		Grp1 = self.flux[:N]/(XS.VEL1*dt)
		Grp2 = self.flux[N:]/(XS.VEL2*dt)
		self.Src[:N] = (fission_term + Grp1 + SumC)*opts.delta
		self.Src[N:] = Grp2*opts.delta

		# New flux
		self.iterations = 0
		while ERRflux > opts.FluxConvError:
			self.iterations += 1
			lastFlux = copy.copy(self.flux)
			self.flux[::2] = (self.Src - O.dot(self.flux))[::2]/D[::2]
			self.flux[1::2] = (self.Src - O.dot(self.flux))[1::2]/D[1::2]
			RMSflux = abs((lastFlux[:]-self.flux[:])/self.flux[:])
			ERRflux = np.linalg.norm(RMSflux,2)/len(self.flux)

		# Update precursors with analytic solution
		mod1[:] = -np.exp(-(XS.DECAYi[:]+wp)*dt) + (1-np.exp(-(XS.DECAYi[:]+wp)*dt))/((XS.DECAYi[:]+wp)*dt)
		mod2[:] = 1 - (1-np.exp(-(XS.DECAYi[:]+wp)*dt))/((XS.DECAYi[:]+wp)*dt)
		PrecursorFactor[:] = XS.Bi[:]*np.exp(wp*dt)/(wp+XS.DECAYi[:])/self.kcrit
		for i in range(2):
			RRfis_mod[i,:] = XS.fis[:N] * (start_flux[:N]*mod1[i] + self.flux[:N]*mod2[i]) + \
			XS.fis[N:] * (start_flux[N:]*mod1[i] + self.flux[N:]*mod2[i])
			self.C[i,:] = self.C[i,:]*np.exp(-XS.DECAYi[i]*dt) + PrecursorFactor[i]*RRfis_mod[i,:]

        # new lines for no-PKE
		# Update power
		RRfis[:] = XS.fis[:N]*self.flux[:N]+XS.fis[N:]*self.flux[N:]
		self.AVGflux = np.average(self.flux)
		subBinPower = RRfis[:]*KAPPA*(opts.delta**2)/2.43
		self.Power = sum(subBinPower)/(78*15**2)

		# Update temperature
		dTdt[:] = RRfis[:]*ALPHA/2.43
		self.T[:] = self.T[:]+dTdt[:]*dt
		self.DopplerFactor[:] = GAMMA*(np.sqrt(self.T[:])-np.sqrt(300))

#end class