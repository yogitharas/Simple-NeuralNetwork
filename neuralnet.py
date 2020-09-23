import numpy as np
#activation function

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid1(x):
	return x*(1-x)


inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])

output = np.array([[0],
	              [1],
	              [1],
	              [0]])

np.random.seed(1)
#weights
syn = 2*np.random.random((3,1)) - 1
print("weights before adjustments: ",syn)

for iter in range(60000):

	input_layer = inputs
    # multiplying inputs * weights 
	output_layer = sigmoid(np.dot(input_layer,syn))
	
	error = output - output_layer
	if(iter% 10000 == 0):
		print("error : ",str(np.mean(np.abs(error))))
	#print("error: ",error)
	adjustments = error * sigmoid1(output_layer)
 
	syn = syn + (input_layer.T.dot(adjustments))

print("weights after the adjustments")
print(syn)
print("actual output")
print(output)
print("outputs after training")
print((output_layer))
