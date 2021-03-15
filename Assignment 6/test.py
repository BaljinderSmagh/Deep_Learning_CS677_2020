import numpy as np
import sys
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import DepthwiseConv2D
from keras.models import load_model

testdata=np.load(sys.argv[1])
testlabel=np.load(sys.argv[2])

testdata=np.rollaxis(testdata,1,4)

testlabel = keras.utils.to_categorical(testlabel, num_classes=10)

model = load_model(sys.argv[3])

#evaluate test accuracy
score=model.evaluate(testdata,testlabel,verbose=0)
accuracy=100*score[1]

#print test accuracy after training
print('Test accuracy:',accuracy)
