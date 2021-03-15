# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 01:16:36 2020

@author: Baljinder Smagh
"""

import numpy as np
import sys

data = open('{}/data.csv'.format(sys.argv[1]))
data=data.readlines()
data = np.array([data[i].strip().split(',') for i in range(1,len(data))])
train_img_file = data[:, 0]
train_label = data[:, 1].astype(np.float32)

data = open('{}/data.csv'.format(sys.argv[2]))
data=data.readlines()
data = np.array([data[i].strip().split(',') for i in range(1,len(data))])
test_img_file = data[:, 0]
test_label = data[:, 1].astype(np.float32)

train = np.array([ np.loadtxt('{}/{}'.format(sys.argv[1],k)) for k in train_img_file])

test = np.array([ np.loadtxt('{}/{}'.format(sys.argv[2],k)) for k in test_img_file])

train_label=np.array([1,-1])
rows=len(train)
c=np.random.rand(2,2)
#c=np.array([[1,-1],[-1,1]])
stride=1
x=int(len(train[0])-len(c)/stride)+1
y=x
z=np.zeros((x,y))

sigmoid=lambda x: 1/(1+np.exp(-x))
obj=0
for k in range(rows):
    for i in range(x):
        for j in range(y):
            value=train[k][i:i+x,j:j+y]*c
            z[i][j]=sigmoid(np.sum(value))
    obj+=np.square(np.mean(z)-train_label[k])
print('obj: {}'.format(obj))


epochs=10000
eta=.01
stop=0
iteration=1

prev_obj=np.inf
while(iteration<epochs):
    delz=np.zeros((z.shape))
    #updating convoution kernel(c)
    for k in range(rows):
        for i in range(x):
            for j in range(y):
                value=train[k][i:i+x,j:j+y]*c
                z[i][j]=sigmoid(np.sum(value))
#        print('z_value: \n',z)
        onearray=np.ones((z.shape))
        z_mat=z*(onearray-z)
#        print('z_mat_value: \n',z_mat)
        for i in range(x):
            for j in range(y):
                delz[i][j]+=(np.mean(z)-train_label[k])*(np.sum(train[k][i:i+x,j:j+y]*z_mat))    
#        print('delz: \n',delz)
#    print('this is mul: \n',eta*delz,'\n')
    c=c-eta*delz
#    print('c value:\n',c)
    obj=0
    #objective
    for k in range(rows):
        for i in range(x):
            for j in range(y):
                value=train[k][i:i+x,j:j+y]*c
                z[i][j]=sigmoid(np.sum(value))
        obj+=np.square(np.mean(z)-train_label[k])
    print('obj: {}\t epoch: {}'.format(obj,iteration))
    iteration+=1
    if prev_obj-obj<0.0001:
        break
    prev_obj=obj
new = c
        
new[np.where(new > np.mean(c))] = 1
new[np.where(new < np.mean(c))] = -1

print(new)
output=[]
for k in range(rows):
        for i in range(x):
            for j in range(y):
                value=train[k][i:i+x,j:j+y]*c
                z[i][j]=sigmoid(np.sum(value))
        output.append(np.mean(z))
output=np.array(output)
new=output
new[np.where(new> np.mean(output))] = 1
new[np.where(new < np.mean(output))] = -1
print('Output: ',new)
count=0
for i in range(len(test_label)):
    if new[i]==test_label[i]:
        count+=1
print('Accuracy:{}%'.format((count/len(test_label))*100))

        


        


    
