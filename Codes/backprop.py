# Backpropogation Neural Network
# Arun Kumar A
# AIMS
# aka.bhagya@gmail.com 
# 15-08-14
# Reference : Ryan's Lectures


import sys
import numpy as np

# Class Definition
class BackpropNN:
	
	# Class Members
	n_layers = 0	# Layer Count
	shape = None	# Tuple that is going to be the exact size of the network
	weights = []	# Weights, list of weights which is numpy array will created
	
	# Class Functions
	def __init__(self,layerSize):	# Layer size will be tuple input
		
		#~~~ Initialize Network ~~~#
		
		# Layer Attributes 
		self.n_layers =  len(layerSize)-1	# "-1" because input layer is not considered 
		self.shape = layerSize
	
		# Input / Output Data from last run
		self._layerInput = []
		self._layerOutput = []
		
		# Create the weight arrays
		for (l1,l2) in zip(layerSize[:-1],layerSize[1:]):
			self.weights.append(np.random.normal(scale= 0.01,size = (l2,l1+1) ))
	
		# Run the network 
		#~~If its the input layer , transfer values from input variable . If its a hidden layer, values are transferred from previous layer.
	def Run(self,input):
		lnCases = input.shape[0] 
		
		self._layerInput = []
		self._layerOutput = []
		
		for index in range(self.n_layers):
		# Determine layer input
			if index == 0: # If input layer
				# First set weights of input layer
				layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1,lnCases])]))
				#      Input to  columns, Fake bias values equaling nmbr of inputs
				
			else:
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,lnCases])]))
				# Take Values of previous list elements
				
			# Save layer inputs
			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sigmoid(layerInput))
			
		# Return Network  output
		return self._layerOutput[-1].T 	# Output is again made a row vector
	
	##################
	# Training Epochs	
	##################
	def TrainEpoch(self,input,target,trainingRate = 0.2):

		delta =[]  # Changes for weights
		lnCases = input.shape[0]

		# Run the network
		self.Run(input)

		# Compute delta values
		""" Since Backpropogation is used we have to start from output layer back to input """
		for index in reversed(range(self.n_layers)):
			# Now we need to decide if the delta is calculated for the output layer or intermediate ones
			if index == self.n_layers - 1 :
				# Compare current output to target
				output_delta = self._layerOutput[index] - target.T 
				error = np.sum(output_delta**2) # Error
				delta.append(output_delta * self.sigmoid(self._layerInput[index],True)) # delta times derivative of sgmd

			else:

				# Compare to following layers delta
				delta_pullback = self.weights[index + 1].T.dot(delta[-1]) # Pull deltas from suceeding layer to current 
				delta.append(delta_pullback[:-1, :] * self.sigmoid(self._layerInput[index],True)) # Last row (-1) is weights of bias

		# Compute weight deltas
		for index in range(self.n_layers):
			 delta_index = self.n_layers - 1 - index 


			 # Get list of layer outputs
			 if index == 0:
			 	layerOutput = np.vstack([input.T , np.ones([1, lnCases])])  # If we are looking at an input layer
			 else:
			 	layerOutput = np.vstack([self._layerOutput[index-1],np.ones([1,self._layerOutput[index-1].shape[1]])])

			 weightDelta = np.sum(layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0), axis=0)

			 self.weights[index] -= trainingRate * weightDelta

		return error




		# Transfer Function
	def sigmoid(self,x,derivative=False):
		if not derivative:
			return 1/(1+np.exp(-x) ) 	# Normal sigmoid function
		else:
			# Derivative of sigmoid  = output * (1-output)
			out = self.sigmoid(x)
			return out * (1-out)
	
	

# If run as test script , create a test object
if __name__== "__main__" :
	bpn = BackpropNN((2,4,1))

	# Print something pretty
	print "~~~~~~~~~~~~"
	print  "Test Stage"
	print "~~~~~~~~~~~~"
	print  "Shape of network is ", bpn.shape
	#print  "Weights of network", bpn.weights
	
	datain = np.array([[0,0],[0,1],[1,0],[1,1] ])
	#data_target = np.array([[0.05],[0.05],[0.95],[0.95]])
	data_target = np.array([[0],[0],[1],[1]])

	lnMax = 100000
	lnerr = 1e-6

	for i in range(lnMax-1):
		err = bpn.TrainEpoch(datain,data_target)

		if i % 10000 == 0:

			print("Iteration {0} \t Error: {1:0.6f}".format(i,err))

		if err < lnerr :
			print("Minimum error reached at iteration {0}".format(i) )
			break

	
	dataout = bpn.Run(datain)
	
	#print "Input : {0} \nOutput : {1}".format(datain,dataout)
	print "Input 	 Output"
	print "=====     ======"
	for i in range(0,np.shape(datain)[0]):
		print datain[i,:],dataout[i]


	
	

