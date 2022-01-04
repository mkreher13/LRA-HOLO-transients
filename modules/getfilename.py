# Class to get input file name

import os
import sys

class FileName:
	
###############################################################

	def __init__(self):
		self
		
###############################################################

	def get_filename(self):
		
		filenames = []
		for i in os.listdir(os.getcwd()):
			if i.endswith(".inp"):
				filenames.append(i)
			else:
				continue
				
		filenames.sort()	
		self.listfn = filenames

#end class