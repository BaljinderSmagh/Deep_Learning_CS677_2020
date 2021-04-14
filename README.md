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

# Assignment 6: CNN

Write a convolutional network in Keras to train the Mini-ImageNet 
dataset on the course website. Your constraint is to create a network
that achieves at least 80% test accuracy (in order to get full points).

# Assignment 7: Transfer Learning

Write a convolutional network in Keras to train the Mini-ImageNet 
dataset on the course website. You may use transfer learning. Your
goal is to achieve above 90% accuracy on the test/validation datasets.

# Assignment 8: Image Classification

Classify images in the three Kaggle datasets Flwoers, Fruits, Chest X-rays 
with convolutional networks. You may use transfer learning. Your
goal is to achieve above 85% accuracy on the test/validation datasets.

# Assignment 9: MNIST GAN

Implement a simple GAN in Keras to generate MNIST images. Use the GAN given here

https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

as your discriminator and generator. 

You want to train the generator to produce images of numbers between 0 and 9.

# Assignment 10: Black-box adversial attack

Implement a simple black box attack in Keras to attack a pretrained 
ResNet18 model from Keras. For the substitute model we use a two hidden 
layer neural network with each layer having 100 nodes.

Our goal is to generate adversaries to decieve a simple single layer 
neural network with 20 hidden nodes into misclassifying data from a 
test set that is provided by us. This test set consists of examples 
from classes 0 and 1 from CIFAR10. 

Your target model should have at least 85% accuracy on the test set without
adversaries. 

A successful attack should have a classification accuracy of at most 10%
on the test.

# Assignment 11: Word2Vec

Learn a word2vec model from fake news dataset and a real news dataset. We 
will use the word2vec model implemented in the Python Gensim library. Now 
we have two sets of word representations learnt from different datasets. 

Output the top 5 most similar words to the following ones from each 
representation.

1. Hillary
2. Trump
3. Obama
4. Immigration
