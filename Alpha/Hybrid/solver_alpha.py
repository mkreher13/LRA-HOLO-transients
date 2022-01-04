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
		self.k = 1.0
		self.a = 0.0

###############################################################

	def gauss_seidel(self, opts, D, O, F, XS):

		# tracemalloc.start()

		# Create local array variables
		N = opts.numBins*opts.numBins
		RMSflux = np.zeros(self.rank)
		RMSsource = np.zeros(self.rank)
		subBinPower = np.zeros(N)
		KAPPA = 3.204e-11 # [J/fission]
		D[:N] = D[:N] + self.a/XS.VEL1
		D[N:] = D[N:] + self.a/XS.VEL2

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
		subBinPower[:] = (XS.fis[:N]*self.flux[:N]+XS.fis[N:]*self.flux[N:])*KAPPA*(opts.delta**2)/2.43
		Power = sum(subBinPower)/(78*15**2)
		Pnorm = opts.ReactorPower/Power
		self.flux = self.flux*Pnorm

###############################################################

	def hill_alg(self, opts, D, O, F, XS):

		# Create local array variables
		N = opts.numBins*opts.numBins
		RMSflux = np.zeros(self.rank)
		RMSsource = np.zeros(self.rank)
		subBinPower = np.zeros(N)
		original_D = copy.copy(D)
		KAPPA = 3.204e-11 # [J/fission]
		evm = 0.01

		# Alpha-eigenvalue outer iteration
		for j in range(1,100):
			last_k = copy.copy(self.k)
			self.k = 1.0
			D[:N] = original_D[:N] + self.a/XS.VEL1
			D[N:] = original_D[N:] + self.a/XS.VEL2

			# k-effective inner iteration
			ERRsource = 1000.
			while ERRsource > opts.FisConvError:

				# Reset local variablbes
				RMSsource[:] = 0
				lastB = copy.copy(self.B)
				self.B = F.dot(self.flux)/self.k

				# Flux iteration
				ERRflux = 1000.
				while ERRflux > opts.FluxConvError:
					lastFlux = copy.copy(self.flux)
					self.flux[::2]  = (self.B - O.dot(self.flux))[::2]/D[::2]
					self.flux[1::2] = (self.B - O.dot(self.flux))[1::2]/D[1::2]
					RMSflux[:] = abs((lastFlux[:]-self.flux[:])/self.flux[:])
					ERRflux = np.linalg.norm(RMSflux, 2)/len(self.flux)
				self.k = sum(F.dot(self.flux))/sum(self.B)

				for i in range(len(self.B)):
					if self.B[i] != 0:
						RMSsource[i] = abs((lastB[i]-self.B[i])/self.B[i])
				ERRsource = np.linalg.norm(RMSsource, 2)/len(self.B)

			if j == 2:
				last_a = copy.copy(self.a)
				self.a = self.a + evm

			elif j > 2:
				if abs(self.a-last_a) < 1e-10:
					break
				self.a = last_a + (1-last_k)/(self.k-last_k)*(self.a-last_a)
				last_a = copy.copy(self.a)

			else:
				continue 

		print("alpha = %.5f" % self.a)
		print("k-eff = %.5f" % self.k)

		# Power normalization
		subBinPower[:] = (XS.fis[:N]*self.flux[:N]+XS.fis[N:]*self.flux[N:])*KAPPA*(opts.delta**2)/2.43
		Power = sum(subBinPower)/(78*15**2)
		Pnorm = opts.ReactorPower/Power
		self.flux = self.flux*Pnorm

#end class