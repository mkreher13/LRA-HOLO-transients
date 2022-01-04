# Class to solve the steady-state problem by power iteration

import copy
import numpy as np
import scipy.linalg
from scipy.sparse import coo_matrix
import time as runtime
import tracemalloc


class SolveSS:
	
###############################################################

	def __init__(self, opts):

		N = opts.numBins*opts.numBins
		self.rank = N*opts.numGroups
		self.flux = np.zeros(self.rank)
		self.B = np.zeros(self.rank)

		# Initial guess 
		self.flux[:] = 1/opts.length
		self.B[:] = 1
		self.k = 1

###############################################################

	def point_jacobi(self, opts, D, O, F, XS):

		# Create local array variables
		N = opts.numBins*opts.numBins
		RMSflux = np.zeros(self.rank)
		RMSsource = np.zeros(self.rank)
		subBinPower = np.zeros(N)

		# Initialize local variables
		j = 0
		m = 0
		ERRsource = 1000.

		# Fission source iteration (outer loop)
		while ERRsource > opts.FisConvError:

			# Reset local variables
			j += 1
			n  = 0
			RMSsource[:] = 0
			lastB  = self.B
			self.B = F.dot(self.flux)/self.k

			# Flux iteration (inner loop)
			ERRflux = 1000.
			while ERRflux > opts.FluxConvError:
				n += 1
				lastFlux  = self.flux
				self.flux = self.B - O.dot(self.flux)
				self.flux[:] = self.flux[:]/D[:]
				RMSflux[:] = abs((lastFlux[:]-self.flux[:])/self.flux[:])
				ERRflux = np.linalg.norm(RMSflux, 2)/len(self.flux)
			m += n

			# Calculate the fission source in each spatial bin
			self.k = sum(F.dot(self.flux))/sum(self.B)

			# Calculate the relative difference in the source between
			# consecutive iterations and take the infinity norm.
			for i in range(len(self.B)):
				if self.B[i] != 0:
					RMSsource[i] = abs((lastB[i]-self.B[i])/self.B[i])
			ERRsource = np.linalg.norm(RMSsource, 2)/len(self.B)

		self.source_it = j
		self.flux_it = m

		# Power normalization
		subBinPower[:] = (XS.fis[:N]*self.flux[:N]+XS.fis[N:]*self.flux[N:])*3.204e-11*(opts.delta**2)/2.43
		Power = sum(subBinPower)/(78*15**2)
		Pnorm = opts.ReactorPower/Power
		self.flux = self.flux*Pnorm

###############################################################

	def gauss_seidel(self, opts, D, O, F, XS):

		# tracemalloc.start()

		# Create local array variables
		N = opts.numBins*opts.numBins
		RMSflux = np.zeros(self.rank)
		RMSsource = np.zeros(self.rank)
		subBinPower = np.zeros(N)

		# Initialize local variables
		j = 0
		m = 0
		ERRsource = 1000.

		# current, peak = tracemalloc.get_traced_memory()
		# print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
		# tracemalloc.stop()
		
		# Fission source iteration (outer loop)
		while ERRsource > opts.FisConvError:

			# Reset local variables
			j += 1
			n  = 0
			RMSsource[:] = 0
			lastB  = copy.copy(self.B)
			self.B = F.dot(self.flux)/self.k
			
			# Flux iteration (inner loop)
			ERRflux = 1000.
			while ERRflux > opts.FluxConvError:
				n += 1
				lastFlux = copy.copy(self.flux)
				self.flux[::2]  = (self.B - O.dot(self.flux))[::2]/D[::2]
				self.flux[1::2] = (self.B - O.dot(self.flux))[1::2]/D[1::2]
				RMSflux[:] = abs((lastFlux[:]-self.flux[:])/self.flux[:])
				ERRflux = np.linalg.norm(RMSflux, 2)/len(self.flux)
			m += n

			# Calculate the fission source in each spatial bin
			self.k = sum(F.dot(self.flux))/sum(self.B)

			# Calculate the relative difference in the source between
			# consecutive iterations and take the infinity norm.
			for i in range(len(self.B)):
				if self.B[i] != 0:
					RMSsource[i] = abs((lastB[i]-self.B[i])/self.B[i])
			ERRsource = np.linalg.norm(RMSsource, 2)/len(self.B)

		self.source_it = j
		self.flux_it = m

		# Power normalization
		subBinPower[:] = (XS.fis[:N]*self.flux[:N]+XS.fis[N:]*self.flux[N:])*3.204e-11*(opts.delta**2)/2.43
		Power = sum(subBinPower)/(78*15**2) #121 or 78?
		Pnorm = opts.ReactorPower/Power
		self.flux = self.flux*Pnorm

#end class