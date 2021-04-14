# Assignment 1
Write a CUDA program for computing the dot product of a vector in parallel with 
each row of a matrix. You are required to have each thread access consecutive
memory locations (coalescent memory access). The inputs are 

1. number of rows
2. number of columns
3. a data matrix file similar to the format in the Chi2 program 
4. a vector file (one row)
5. cuda device
6. number of threads

For example if the input is

1 2 0
1 1 0
1 2 1

and w = (2, 4, 6)

then your program should output

10
6
16

Compute the dot products in parallel your kernel function. You will have to
transpose the data matrix in order to get coalescent memory access. 

# Assigment 2
Convert the CUDA program written in assignment one into an
OpenMP one. The output of both your CUDA and OpenML programs must be the same. 

# Assignment 3
Write a Python program that trains a single layer neural network
with sigmoid activation. You may use numpy. Your input is in dense 
liblinear format which means you exclude the dimension and include 0's. 

Let your program command line be:

python single_layer_nn.py <train> <test> <n>

where n is the number of nodes in the single hidden layer.

For this assignment you basically have to implement gradient
descent. Use the update equations we derived on our google document
shared with the class.

Test your program on the XOR dataset:

1 0 0
1 1 1
-1 0 1
-1 1 0

1. Does your network reach 0 training error? 

2. Can you make your program into stochastic gradient descent (SGD)?

3. Does SGD give lower test error than full gradient descent?


# Assignment 4

Implement stochastic gradient descent in your back propagation program
in assignment 3. We will do the mini-batch SGD search. 

I. Mini-batch SGD algorithm:

Initialize random weights
for(k = 0 to n_epochs):
	Shuffle the rows (or row indices)
	for j = 0 to rows-1:
		Select the first k datapoints where k is the mini-batch size
		Determine gradient using just the selected k datapoints
		Update weights with gradient
	Recalculate objective

Your input, output, and command line parameters are the same as assignment 3.
We take the batch size k as input. We leave the offset for the final layer 
to be zero at this time.
