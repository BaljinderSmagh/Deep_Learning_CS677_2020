import sys
import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer,GlobalAveragePooling2D
from keras.models import Sequential
from keras import optimizers
from keras import applications
from keras import backend as K

def relu6(x):
  return K.relu(x, max_value=6)

testdata=sys.argv[1]

test_datagen = ImageDataGenerator(
        rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        directory=testdata,
        target_size=(32,32),
        batch_size=64,
        class_mode='categorical')

from keras.utils.generic_utils import CustomObjectScope

with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
	model = load_model(sys.argv[2])
filenames = test_generator.filenames
nb_samples = len(filenames)
predict=model.evaluate_generator(test_generator,nb_samples,verbose=0)

accuracy = 100 * predict[1]

#print test accuracy after training
print('Test accuracy:',accuracy)
