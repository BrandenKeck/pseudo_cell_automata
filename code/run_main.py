'''
This Script is a RUN function which uses the cellular automation defined in 'biosystem.py' to classify data from the popular Iris Flower dataset.  Error between predicted results is then calculated and compared to other models.
'''

# Standard Imports
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Classifier Imports
import biosystem
from biosystem import culture, cell
from sklearn.svm import SVC
from sklearn import tree

if __name__ == "__main__":
	
	# Ignore warnings
	import warnings
	warnings.simplefilter("ignore")
	
	# Start tracking time
	import time
	start_me = time.time()
	
	# Print header
	print("")
	print("#############################################")
	print("  Pseudo Cellular Automation Classification  ")
	print("           Author: Branden Keck              ")
	print("#############################################")
	print("")
	
	# Important simulation variables
	m = 12
	iter = 50
	
	print("Initializing Cell Culture...")
	print("")
	
	# Create the model
	luca = culture([m, m, m, m])
	
	# Import the data from sklearn and shuffle it into training and testing datasets
	print("Importing Iris Dataset...")
	print("")
	XT, YT = load_iris(return_X_y = True)
	X_train, X_test, Y_train, Y_test = train_test_split(XT, YT, test_size=0.5)
	
	# Initialize model data
	print("Inoculating Cells...")
	print("")
	luca.inoculate(X_train, Y_train)
	
	# Iteration cycles
	print("Training...  Printing Totals...")
	print("")
	for i in range(iter):
		a = luca.ferment()
		print(a)
	
	print("")
	print("Creating Predictions...")
	print("")
	
	# Draw predictions
	Y0 = luca.harvest(X_test)
	correct = 0
	for i in range(Y0.size):
		if Y0[i] == Y_test[i]:
			correct += 1
	pCell = correct/Y0.size
	
	# Train and predict with SVM
	svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
	svm.fit(X_train, Y_train)
	pSVM = svm.score(X_test, Y_test)
	
	# Train and predict with Decision Tree
	dt = tree.DecisionTreeClassifier(criterion='gini')
	dt.fit(X_train, Y_train)
	pDT = dt.score(X_test, Y_test)
	
	# Print results
	print("")
	print("RESULTS:")
	print("")
	print("Training Set Size: " + str(Y_train.size))
	print("Test Set Size: " + str(Y_test.size))
	print("")
	print("Pseudo-Cellular Automation: " + str(pCell))
	print("Support Vector Machine: " + str(pSVM))
	print("Decision Tree: " + str(pDT))
	print("")
	
	# Print run time
	end_me = time.time() - start_me
	print("Run Time:")
	print(end_me)