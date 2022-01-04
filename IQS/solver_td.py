# Class to solve the time-dependent problem
# in an improved quasi-static scheme

import copy
import numpy as np
from scipy.sparse import coo_matrix


class SolveTD:

###############################################################
	
	def __init__(self, opts, flux, C, kcrit):

		N = opts.numBins*opts.numBins
		self.flux = flux
		self.Src  = np.zeros(N*opts.numGroups)
		self.DopplerFactor = np.zeros(N)
		self.kcrit = kcrit
		self.C = C
		
###############################################################

	def gauss_seidel(self, opts, amp, D, O, XS, time, wd1, wd2, wp):

		# Initialize local variables
		dt = opts.dt
		N  = opts.numBins*opts.numBins
		RRfis = np.zeros([N])
		dTdt  = np.zeros(N)
		ERRflux = 1000.
		count = 0
		wd = [wd1, wd2]

		# Update S vector
		SumC = np.dot(XS.DECAYi,self.C)/amp	
		Grp1 = self.flux[:N]/(XS.VEL1*dt)
		Grp2 = self.flux[N:]/(XS.VEL2*dt)
		self.Src[:N] = (Grp1 - SumC)*opts.delta
		self.Src[N:] = Grp2*opts.delta

		# New flux & precusors
		self.iterations = 0
		while ERRflux > opts.FluxConvError:
			self.iterations += 1
			lastFlux = copy.copy(self.flux)
			self.flux[::2] = (self.Src - O.dot(self.flux))[::2]/D[::2]
			self.flux[1::2] = (self.Src - O.dot(self.flux))[1::2]/D[1::2]
			RMSflux = abs((lastFlux[:]-self.flux[:])/self.flux[:])
			ERRflux = np.linalg.norm(RMSflux,2)/len(self.flux)