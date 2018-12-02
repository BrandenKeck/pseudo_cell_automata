'''
This Script is NOT intended to be run on its own - it should be called from run_2d_comparison.py

This Script is used for the purposes of calculating classifiers based on a completely non-biased VON NEUMANN Cellular Automation.  The results of this "simple" method are used to determine if adding complexity to cell growth rules is beneficial to overall classification results.
'''

# Standard Imports
import numpy as np
import operator
from copy import deepcopy

# Cell Class - Used only for storing data properties
class cell():
	
	def __init__(self, s):
		self.species = s

# Primary Classifier Class
class culture():
	
	# Initialize important Properties
	def __init__(self, d):
		
		self.bins = []
		self.dimensions = d
		self.cells = empties(d)
	
	# Fill initial cells based on data values
	def inoculate(self, data, classes):
		
		# dimension variables
		dims = np.array(self.dimensions)
		n = len(self.dimensions)

		# Creation of bins
		for j in range(0,n):
			
			min_dat = np.min(data[:, j]) - 0.1*(np.max(data[:, j])-np.min(data[:, j]))
			max_dat = np.max(data[:, j]) + 0.1*(np.max(data[:, j])-np.min(data[:, j]))
			delta = (max_dat-min_dat)/dims[j]
			
			self.bins.append(np.arange(min_dat, max_dat, delta)+delta)
		
		# Sorting of data into bins
		for i, r in enumerate(data):

			idxs = []
			for j, c in enumerate(r):
				idxs.append(np.argmax(c <= self.bins[j]))

			self.reproduce(idxs, cell(classes[i]))
		
		# Competition step needed to ensure there is only one cell per bin
		# Species with max cells in neighborhood is given control of the bin
		self.compete()

	# Iteration function - cell growth
	def ferment(self):
		
		# Create dictionary to track cell totals
		abundance = dict()
		
		# Iteration variables
		n = len(self.dimensions)
		c0 = deepcopy(self.cells)
		dims = np.array(self.dimensions)
		frmnt = np.zeros(dims.shape).astype(int)
		
		# Loop through every cell
		done = False
		while not done:
			
			# Get cell array for the current index - check if empty
			cel = self.get_cell(c0, frmnt)
			if cel:
			
				# Update the abundances
				if cel[0].species in abundance:
					abundance[cel[0].species] += 1
				else:
					abundance[cel[0].species] = 1
				
				# Add cells to von Neumann neighbors
				for d in range(0,dims.shape[0]):
					if frmnt[d] > 0:
						
						instance = deepcopy(frmnt)
						instance[d] -= 1
						
						if not self.get_cell(c0, instance):
							self.reproduce(instance, cell(cel[0].species))

					if frmnt[d] < dims[d]-1:
						
						instance = deepcopy(frmnt)
						instance[d] += 1
						
						if not self.get_cell(c0, instance):
							self.reproduce(instance, cell(cel[0].species))
			
			# Increment cell index
			frmnt = increment(frmnt, dims)
			if 0 in np.where(dims-frmnt == 0)[0]: done = True
		
		# Competition step needed to ensure there is only one cell per bin
		# Species with max cells in neighborhood is given control of the bin
		self.compete()
		
		# Return new cell totals
		return abundance

	# Competition function - ensure one cell per space on the grid
	def compete(self):
		
		# Iteration variables
		dims = np.array(self.dimensions)
		cmpt = np.zeros(dims.shape).astype(int)
		
		# Loop through all cells
		done = False
		while not done:
			
			# Check if cell array empty
			cel = self.get_cell(self.cells, cmpt)
			if cel:
				
				# Calculate species with maximum cell count
				competitors = dict()
				for c in cel:
					if c.species in competitors:
						competitors[c.species] += 1
					else:
						competitors[c.species] = 1
				
				cel[:] = []
				
				# Cell becomes the resulting species
				spec = max(competitors.items(), key=operator.itemgetter(1))[0]		
				cel.append(cell(spec))
			
			# Increment the index
			cmpt = increment(cmpt, dims)
			if 0 in np.where(dims-cmpt == 0)[0]: done = True
	
	# Function to add cells to cell array at a given index
	# Catch common location error
	def reproduce(self, pos, daughter):
	
		location = self.cells
		for p in pos:
			try:
				location = location[p]
			except:
				print("Location Error - Please Restart Simulation")
				quit()
		
		location.append(daughter)
	
	# Function to get cell array at a given index
	def get_cell(self, c0, pos):

		c = c0
		for i in pos:
			c = c[i]
	
		return c
		
# Create array of empties for initial cell grid
def empties(b):
	
	invB = np.flip(b, axis=0)
	empty = []
	for b in invB:
		build = deepcopy(empty)
		empty = []
		for i in range(0,b):
			empty.append(build)

	return np.array(empty).tolist()

# Increment index counter function
# Needed because number of dimensions is unknown
def increment(cntr, dims):

	cntr[cntr.shape[0]-1] += 1
	zeros = False
	if np.where(dims-cntr == 0)[0] != 0: zeros = True
	while zeros:
		idx = np.where(dims-cntr == 0)[0][0]
		cntr[idx] = 0
		cntr[idx-1] += 1
		if np.where(dims-cntr == 0)[0] != 0: zeros = True
		else: zeros = False
		
	return cntr
	
	
	
	