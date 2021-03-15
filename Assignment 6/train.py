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
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

traindata=np.load(sys.argv[1])
trainlabel=np.load(sys.argv[2])

#Rollaxis
traindata=np.rollaxis(traindata,1,4)
trainlabel = keras.utils.to_categorical(trainlabel, num_classes=10)

#data division for cross-validation
length = round(len(traindata)*0.90)

x_train = traindata[:length]
y_train = trainlabel[:length]
x_valid = traindata[length:]
y_valid = trainlabel[length:]

# Initialising the CNN
classifier = Sequential()

classifier.add(DepthwiseConv2D(kernel_size=(3,3), padding='Same',input_shape = (112,112,3), activation = 'relu'))

# 1st Convolution
classifier.add(Conv2D(filters=16, kernel_size=(6, 6), padding='Same',input_shape = (112,112,3), activation = 'relu'))

# Pooling
#classifier.add(AveragePooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))
#  second convolutional layer
classifier.add(Conv2D(filters=32, kernel_size=(3, 3),padding='Same', activation = 'relu'))
classifier.add(AveragePooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization(momentum=0.25))
classifier.add(Dropout(0.25))
# third convolutional layer
classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation = 'relu'))
classifier.add(AveragePooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

# Adding a fourth convolutional layer
classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation = 'relu'))
classifier.add(AveragePooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))
# Adding a fifth convolutional layer

classifier.add(Conv2D(filters=128, kernel_size=(3, 3), activation = 'relu'))
classifier.add(AveragePooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))
#globalpooling
classifier.add(GlobalAveragePooling2D())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(rate=0.35))
classifier.add(Dense(units = 10, activation = 'softmax'))




#optimiser
opt=keras.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999)
classifier.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])


#checkpoint
checkpointer=ModelCheckpoint(filepath=sys.argv[3],verbose=1,save_best_only=True)

classifier.fit(x_train,y_train,
                         batch_size=20,
                         epochs = 50, 
                         validation_data = (x_valid,y_valid),callbacks=[checkpointer],shuffle=True
                        )

classifier.load_weights(sys.argv[3])                       
classifier.save(sys.argv[3])

