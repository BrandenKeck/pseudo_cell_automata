'''
This is the primary script used to conduct simulations for this project.  It should not be run directly.  This script should be called from run_main.py, run_2d_animation.py, or run_2d_comparision.py

The purpose of this script is to conduct pseudo-cellular automata that produce classification boundaries using a variety of principals including cell "fitness" (mixture of alive [1] and dead[0] states), Coulomb repulsion, and use of gradients in cell growth.

The goal is to determine if this set of cell growth rules can be used to create accurate classifiers and provide insight to the underlying probabilities associated with a given dataset
'''

# Standard Imports
import numpy as np
import operator
from copy import deepcopy

# Cell Class - Used only for storing data properties
class cell():
	
	def __init__(self, s, f):
		self.species = s
		self.fitness = f

# Primary Classifier Class
class culture():
	
	# Initialize important Properties
	def __init__(self, d):
		
		self.bins = []
		self.lowerbounds = []
		self.dimensions = d
		self.cells = empties(d)
		self.totalfitness = 0
		self.fitnessdict = dict()
	
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
			self.lowerbounds.append(min_dat)
		
		# Initial sorting of data into bins
		for i, r in enumerate(data):

			idxs = []
			for j, c in enumerate(r):
				idxs.append(np.argmax(c <= self.bins[j]))

			self.reproduce(idxs, cell(classes[i], 1))
		
		# Cells compete for control of the bins
		self.compete()
		
		# Create useful variables for initial cell growth step
		cd, cellcount = self.cell_dictionaries()
		v45 = np.ones(n)/np.sqrt(2)
		
		# Iterate through all cells
		# Perform initial cell growth by adding cells to neighboring spaces on the grid
		for i in range(0, len(cd)):
			di = cd[i]
			
			# If only one cell in a class, growth cannot be based on Coulomb repulsion
			# Add cells with equal fitness to each Moore neighbor
			if cellcount[di["species"]] <= 1:
				for d1 in range(0,dims.shape[0]):
					if di["idx"][d1] < dims[d1]-1:
						instance1 = deepcopy(di["idx"])
						instance1[d1] += 1
						self.reproduce(instance1, cell(di["species"], di["fitness"]/(n**3-1)))
						for d2 in range(0,dims.shape[0]):
							if d2 != d1:
								if instance1[d2] < dims[d2]-1:
									instance2 = deepcopy(instance1)
									instance2[d2] += 1
									self.reproduce(instance2, cell(di["species"], di["fitness"]/(n**3-1)))
								if instance1[d2] > 0:
									instance2 = deepcopy(instance1)
									instance2[d2] -= 1
									self.reproduce(instance2, cell(di["species"], di["fitness"]/(n**3-1)))
					if di["idx"][d1] > 0:
						instance1 = deepcopy(di["idx"])
						instance1[d1] -= 1
						self.reproduce(instance1, cell(di["species"], di["fitness"]/(n**3-1)))
						for d2 in range(0,dims.shape[0]):
							if d2 != d1:
								if instance1[d2] < dims[d2]-1:
									instance2 = deepcopy(instance1)
									instance2[d2] += 1
									self.reproduce(instance2, cell(di["species"], di["fitness"]/(n**3-1)))
								if instance1[d2] > 0:
									instance2 = deepcopy(instance1)
									instance2[d2] -= 1
									self.reproduce(instance2, cell(di["species"], di["fitness"]/(n**3-1)))
			
			else:
				# Calculate a replusion vector based on "Coulomb"-like force equation
				repulsion = np.zeros(n)
				for j in range(0, len(cd)):
					dj = cd[j]
					
					if i!=j and di["species"] == dj["species"]:
						repulsion += (di["fitness"]*dj["fitness"]*(di["idx"] - dj["idx"]))/(np.linalg.norm(di["idx"] - dj["idx"])**2)
				
				# Normalize repulsion vector to a unit vector
				repulsion = repulsion / np.linalg.norm(repulsion)
				
				# Add cells to Moore neighbors that are in the direction of the repulsion vector
				# Total fitness of the cells added to neighboring spaces is equal to the fitness of the current cell
				# Proportion of fitness added to neighboring cells is determined by the angle of the repulsion vector
				for d1 in range(0,dims.shape[0]):
					if repulsion[d1] > 0 and di["idx"][d1] < dims[d1]-1:
						instance1 = deepcopy(di["idx"])
						instance1[d1] += 1
						self.reproduce(instance1, cell(di["species"], di["fitness"]*np.absolute(repulsion[d1])))
						for d2 in range(0,dims.shape[0]):
							if d2 != d1:
								if repulsion[d2] > 0 and instance1[d2] < dims[d2]-1:
									instance2 = deepcopy(instance1)
									instance2[d2] += 1
									self.reproduce(instance2, cell(di["species"], di["fitness"]*(np.absolute(repulsion).dot(v45)/v45.dot(v45))/np.sqrt(2)))
								if repulsion[d2] < 0 and instance1[d2] > 0:
									instance2 = deepcopy(instance1)
									instance2[d2] -= 1
									self.reproduce(instance2, cell(di["species"], di["fitness"]*(np.absolute(repulsion).dot(v45)/v45.dot(v45))/np.sqrt(2)))
					if repulsion[d1] < 0 and di["idx"][d1] > 0:
						instance1 = deepcopy(di["idx"])
						instance1[d1] -= 1
						self.reproduce(instance1, cell(di["species"], di["fitness"]*np.absolute(repulsion[d1])))
						for d2 in range(0,dims.shape[0]):
							if d2 != d1:
								if repulsion[d2] > 0 and instance1[d2] < dims[d2]-1:
									instance2 = deepcopy(instance1)
									instance2[d2] += 1
									self.reproduce(instance2, cell(di["species"], di["fitness"]*(np.absolute(repulsion).dot(v45)/v45.dot(v45))/np.sqrt(2)))
								if repulsion[d2] < 0 and instance1[d2] > 0:
									instance2 = deepcopy(instance1)
									instance2[d2] -= 1
									self.reproduce(instance2, cell(di["species"], di["fitness"]*(np.absolute(repulsion).dot(v45)/v45.dot(v45))/np.sqrt(2)))
		
		# Clear dictionaries from memory
		cd.clear()
		cellcount.clear()
		
		# Cells compete for control of the bins
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
				
				# Calculate a replusion vector based on "Coulomb"-like force equation
				# Computation only considers von Neumann neighbors in this case
				repulsion = np.zeros(n)
				for d in range(0,dims.shape[0]):
					
					instance = deepcopy(frmnt)
					if instance[d] > 0:
						instance[d] -= 1
						repcell = self.get_cell(c0, instance)
						if repcell:
							if repcell[0].species == cel[0].species:
								repulsion += (cel[0].fitness*repcell[0].fitness*(frmnt - instance))
							
					instance = deepcopy(frmnt)
					if instance[d] < dims[d]-1:
						instance[d] += 1
						repcell = self.get_cell(c0, instance)
						if repcell:
							if repcell[0].species == cel[0].species:
								repulsion += (cel[0].fitness*repcell[0].fitness*(frmnt - instance))
				
				# Normalize repulsion vector to a unit vector
				try:
					repulsion = repulsion / np.linalg.norm(repulsion)
				except:
					repulsion = np.zeros(n)

				# Add cells to von Neumann neighbors that are in the direction of the repulsion vector
				# Proportion of fitness added to neighboring cells is determined by the angle of the repulsion vector as well as the gradient of the opposite von Neumann neighbor and the current cell
				for d in range(0,dims.shape[0]):
					if repulsion[d] > 0 and frmnt[d] > 0 and frmnt[d] < dims[d]-1:
						
						instance = deepcopy(frmnt)
						instance[d] += 1
						
						repcell = deepcopy(frmnt)
						repcell[d] -= 1
						rep = self.get_cell(c0, repcell)
						
						try:
							self.reproduce(instance, cell(cel[0].species, np.absolute(repulsion[d])*(cel[0].fitness)/(rep[0].fitness)))
						except:
							self.reproduce(instance, cell(cel[0].species, 0))
						
					if repulsion[d] < 0 and frmnt[d] > 0 and frmnt[d] < dims[d]-1:
						instance = deepcopy(frmnt)
						instance[d] -= 1
						
						repcell = deepcopy(frmnt)
						repcell[d] += 1
						rep = self.get_cell(c0, repcell)
						
						try:
							self.reproduce(instance, cell(cel[0].species, np.absolute(repulsion[d])*(cel[0].fitness)/(rep[0].fitness)))
						except:
							self.reproduce(instance, cell(cel[0].species, 0))

			# Increment cell index
			frmnt = increment(frmnt, dims)
			if 0 in np.where(dims-frmnt == 0)[0]: done = True
		
		# Cells compete for control of the bins
		self.compete()
		
		# Return new cell totals
		return abundance

	# Competition function - ensure one cell per space on the grid
	def compete(self):
		
		# Tracking of overall fitnesses
		self.totalfitness = 0
		self.fitnessdict = dict()
		
		# Dimension variables
		dims = np.array(self.dimensions)
		cmpt = np.zeros(dims.shape).astype(int)
		
		# Loop through all cells
		done = False
		while not done:
			
			# Check if cell array empty
			cel = self.get_cell(self.cells, cmpt)
			if cel:
				
				# Calculate species with maximum overall fitness in the bin
				competitors = dict()
				for c in cel:
					if c.species in competitors:
						competitors[c.species] += c.fitness
					else:
						competitors[c.species] = c.fitness
				
				cel[:] = []
				
				# Fitnesses of other species subtracted out
				spec = max(competitors.items(), key=operator.itemgetter(1))[0]
				fit = 2*competitors[spec] - sum(competitors.values())
				
				# Replace group of cells with appropriate cell
				cel.append(cell(spec, fit))

				# Update variables
				self.totalfitness += fit
				if spec in self.fitnessdict:
					self.fitnessdict[spec] += fit
				else:
					self.fitnessdict[spec] = fit
			
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
	
	# Function to predict classification after training the cell grid
	def harvest(self, data):
		
		ness = []
		for i, r in enumerate(data):
			idxs = []
			for j, c in enumerate(r):
				idxs.append(np.argmax(c <= self.bins[j]))
			
			paula = self.get_cell(self.cells, idxs)
			if paula:
				ness.append(paula[0].species)
			else:
				print("ERROR: Grid is not full, some observations have no mapping.")
				ness.append(-1)
			
		return np.array(ness)
			
	# Function to get cell array at a given index
	def get_cell(self, c0, pos):

		c = c0
		for i in pos:
			c = c[i]
	
		return c
		
	# Construct cell dictionaries
	# Not used often - but an efficient way to represent overall data in a more accessible way
	def cell_dictionaries(self):
		
		cd = dict()
		counts = dict()
			
		dims = np.array(self.dimensions)
		clldct = np.zeros(dims.shape).astype(int)
		
		done = False
		while not done:
			
			cel = self.get_cell(self.cells, clldct)
			if cel:
				cd[len(cd)] = {'idx': deepcopy(clldct), 'species': cel[0].species, 'fitness': cel[0].fitness}
				
				if cel[0].species in counts:
					counts[cel[0].species] += 1
				else:
					counts[cel[0].species] = 1	
			
			clldct = increment(clldct, dims)
			if 0 in np.where(dims-clldct == 0)[0]: done = True
	
		return cd, counts
		
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
	
	
	
	