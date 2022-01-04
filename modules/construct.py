# Class to construct linear system of equations

import numpy as np
from scipy.sparse import coo_matrix


class Construct():
	
###############################################################
	
	def __init__(self):
		self
		
###############################################################
		
	def construct_ss(self, opts, D, rXS, Gscat, fisXS):

		nGrps = opts.numGroups
		nBins = opts.numBins
		N = nBins*nBins
		rank = N*nGrps
		delta = opts.delta

		# D stores the diagonal entries
		self.D = np.zeros(rank)
		# O stores the off-diagonal entries
		data = []
		row_indices = []
		col_indices = []
		# F stores the fission entries
		F_data = []
		F_row_indices = []
		F_col_indices = []

		self.dLeft = np.zeros(rank)
		self.dRight = np.zeros(rank)
		self.dUp = np.zeros(rank)
		self.dDown = np.zeros(rank)

		for g in range(nGrps):
			for y in range(nBins):
				for x in range(nBins):
					i = g*N+x+(y*nBins)

					# Calculate the average diffusion theory between current cell and the adjacent cell
					# and account for boundary conditions

					if x == 0:  # Reflective BC
						self.dLeft[i] = 0 
					else:
						DAdj = D[i-1]
						deltaAdj = delta
						self.dLeft[i] = (2*D[i]*DAdj)/(deltaAdj*D[i]+delta*DAdj)

					if x == (nBins-1):  # Zero flux BC 
						DAdj = 1
						deltaAdj = 0
					else:
						DAdj = D[i+1]
						deltaAdj = delta
					self.dRight[i] = (2*D[i]*DAdj)/(deltaAdj*D[i]+delta*DAdj) 

					if y == 0:  # Reflective BC
						self.dDown[i] = 0 
					else:
						DAdj = D[i-nBins]
						deltaAdj = delta
						self.dDown[i] = (2*D[i]*DAdj)/(deltaAdj*D[i]+delta*DAdj) 

					if y == (nBins-1):  # Zero flux BC
						DAdj = 1
						deltaAdj = 0
					else:
						DAdj = D[i+nBins]
						deltaAdj = delta
					self.dUp[i] = (2*D[i]*DAdj)/(deltaAdj*D[i]+delta*DAdj) 

					if x != 0:
						data.append(-self.dLeft[i])
						row_indices.append(i)
						col_indices.append(i-1)
					if x != (nBins-1):
						data.append(-self.dRight[i])
						row_indices.append(i)
						col_indices.append(i+1)
					if y != 0:
						data.append(-self.dDown[i])
						row_indices.append(i)
						col_indices.append(i-nBins)
					if y != (nBins-1):
						data.append(-self.dUp[i])
						row_indices.append(i)
						col_indices.append(i+nBins)

					# Include downscattering
					if g == 0:
						data.append(-Gscat[x+(y*nBins),1]*delta)
						row_indices.append(i+N)
						col_indices.append(i)

					# Fision Matrix
					if g == 0:
						F_data.append(fisXS[i]*delta)
						F_row_indices.append(i)
						F_col_indices.append(i)
						F_data.append(fisXS[i+N]*delta)
						F_row_indices.append(i)
						F_col_indices.append(i+N)

		self.D[:] = self.dLeft[:]+self.dRight[:]+self.dUp[:]+self.dDown[:]+rXS[:]*delta
		self.O = coo_matrix((data, (row_indices, col_indices)), [rank,rank])
		self.F = coo_matrix((F_data, (F_row_indices, F_col_indices)), [rank,rank])

		self.Otrans = self.O.transpose()
		self.Ftrans = self.F.transpose()

###############################################################
		
	def construct_td(self, opts, D, PrXS, Gscat, PfisXS):

		nGrps = opts.numGroups
		nBins = opts.numBins
		N = nBins*nBins
		rank = N*nGrps
		delta = opts.delta

		# D stores the diagonal entries
		self.D = np.zeros(rank)
		# O stores the off-diagonal entries
		data = []
		row_indices = []
		col_indices = []

		for g in range(0, nGrps):
			for y in range(nBins):
				for x in range(nBins):
					i = g*N+x+(y*nBins)

					if g == 0:
						F1 = PfisXS[i]*delta
						data.append(-PfisXS[i+N]*delta)
						row_indices.append(i)
						col_indices.append(i+N)
					else:
						F1 = 0
						F2 = 0

					self.D[i] = self.dLeft[i]+self.dRight[i]+self.dUp[i]+self.dDown[i]+PrXS[i]*delta-F1

					if x != 0:
						data.append(-self.dLeft[i])
						row_indices.append(i)
						col_indices.append(i-1)
					if x != (nBins-1):
						data.append(-self.dRight[i])
						row_indices.append(i)
						col_indices.append(i+1)
					if y != 0:
						data.append(-self.dDown[i])
						row_indices.append(i)
						col_indices.append(i-nBins)
					if y != (nBins-1):
						data.append(-self.dUp[i])
						row_indices.append(i)
						col_indices.append(i+nBins)

					# Include downscattering
					if g == 0:
						data.append(-Gscat[x+(y*nBins),1]*delta)
						row_indices.append(i+N)
						col_indices.append(i)

		self.O = coo_matrix((data, (row_indices, col_indices)), [rank,rank])

###############################################################
		
	def update_ss(self, opts, rXS):

		delta = opts.delta
		self.D[:] = self.dLeft[:]+self.dRight[:]+self.dUp[:]+self.dDown[:]+rXS[:]*delta

###############################################################

	def update_td(self, opts, XS):

		nBins = opts.numBins
		N = nBins*nBins
		delta = opts.delta

		self.D[:N] = self.dLeft[:N]+self.dRight[:N]+self.dUp[:N]+self.dDown[:N]+XS.Premoval[:N]*delta-XS.Pfis[:N]*delta
		self.D[N:] = self.dLeft[N:]+self.dRight[N:]+self.dUp[N:]+self.dDown[N:]+XS.Premoval[N:]*delta

###############################################################
		
	def update_freq(self, opts, XS, kcrit, wp, wd1, wd2):

		nGrps = opts.numGroups
		nBins = opts.numBins
		N = nBins*nBins
		rank = N*nGrps
		delta = opts.delta
		wd = [wd1, wd2]

		Rfactor = [wp/XS.VEL1, wp/XS.VEL2]

		Ffactor = (1-XS.B_TOT)/kcrit+sum(
			(XS.Bi[:]*XS.DECAYi[:])/(wd[:]+XS.DECAYi[:])/kcrit)

		F_data = []
		F_row_indices = []
		F_col_indices = []

		for y in range(nBins):
			for x in range(nBins):
				i = x+(y*nBins)

				F_data.append(XS.fis[i]*Ffactor*delta)
				F_row_indices.append(i)
				F_col_indices.append(i)
				F_data.append(XS.fis[i+N]*Ffactor*delta)
				F_row_indices.append(i)
				F_col_indices.append(i+N)

		self.D[:N] = self.dLeft[:N]+self.dRight[:N]+self.dUp[:N]+self.dDown[:N]+(XS.removal[:N]+Rfactor[0])*delta
		self.D[N:] = self.dLeft[N:]+self.dRight[N:]+self.dUp[N:]+self.dDown[N:]+(XS.removal[N:]+Rfactor[1])*delta

		self.F = coo_matrix((F_data, (F_row_indices, F_col_indices)), [rank,rank])
		self.Ftrans = self.F.transpose()

#end class

###############################################################
		
	def update_spatial_freq(self, opts, XS, kcrit, wp, wd1, wd2):

		nGrps = opts.numGroups
		nBins = opts.numBins
		N = nBins*nBins
		rank = N*nGrps
		delta = opts.delta

		Rfactor = [wp/XS.VEL1, wp/XS.VEL2]

		Ffactor = np.zeros(N)
		Ffactor[:] = (1-XS.B_TOT)/kcrit + (XS.Bi[0]*XS.DECAYi[0])/(wd1[:]+XS.DECAYi[0])/kcrit + (XS.Bi[1]*XS.DECAYi[1])/(wd2[:]+XS.DECAYi[1])/kcrit

		F_data = []
		F_row_indices = []
		F_col_indices = []

		for y in range(nBins):
			for x in range(nBins):
				i = x+(y*nBins)

				F_data.append(XS.fis[i]*Ffactor[i]*delta)
				F_row_indices.append(i)
				F_col_indices.append(i)
				F_data.append(XS.fis[i+N]*Ffactor[i]*delta)
				F_row_indices.append(i)
				F_col_indices.append(i+N)

		self.D[:N] = self.dLeft[:N]+self.dRight[:N]+self.dUp[:N]+self.dDown[:N]+(XS.removal[:N]+Rfactor[0])*delta
		self.D[N:] = self.dLeft[N:]+self.dRight[N:]+self.dUp[N:]+self.dDown[N:]+(XS.removal[N:]+Rfactor[1])*delta

		self.F = coo_matrix((F_data, (F_row_indices, F_col_indices)), [rank,rank])
		self.Ftrans = self.F.transpose()

#end class