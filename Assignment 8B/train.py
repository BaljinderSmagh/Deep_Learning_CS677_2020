import sys
import keras
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input,Dense, Dropout, InputLayer,GlobalAveragePooling2D,BatchNormalization
from keras.models import Sequential
from keras import optimizers
from keras.models import load_model

train_folder=('{}/train'.format(sys.argv[1]))
train_data=train_folder
validation_data=('{}/test'.format(sys.argv[1]))
tensor=Input(shape=(224,224,3))
base_inception = applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, 
                             input_tensor=tensor)
                             
# Add a global spatial average pooling layer
out = base_inception.output
out = Dropout(0.5)(out)
out = GlobalAveragePooling2D()(out)
out = Dense(128, activation='relu')(out)
out = Dropout(0.2)(out)
out = Dense(128,kernel_regularizer=regularizers.l2(0.001), activation='relu')(out)
out = Dense(64,kernel_regularizer=regularizers.l2(0.001), activation='relu')(out)
#out = Dense(64,kernel_regularizer=regularizers.l2(0.0001), activation='relu')(out)
#out = Dense(32,kernel_regularizer=regularizers.l2(0.0001), activation='relu')(out)
#out = Dense(32,kernel_regularizer=regularizers.l2(0.0001), activation='relu')(out)
#out = Dense(16,kernel_regularizer=regularizers.l2(0.0001), activation='relu')(out)
#out = Dense(16,kernel_regularizer=regularizers.l2(0.0001), activation='relu')(out)
out  = BatchNormalization()(out)
total_classes = 2
predictions = Dense(total_classes, activation='softmax')(out)

model = Model(inputs=base_inception.input, outputs=predictions)

# only if we want to freeze layers
for layer in base_inception.layers:
    layer.trainable = False


opt=keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


checkpointer=ModelCheckpoint(filepath=sys.argv[2],verbose=1,save_best_only=True)


train_datagen = ImageDataGenerator(
        rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        directory=train_data,
        target_size=(224,224),
        batch_size=64,
        class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(directory=validation_data,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')


model.fit_generator(
        train_generator,
        steps_per_epoch=80, epochs=20,
        validation_data=validation_generator, validation_steps=40,callbacks=[checkpointer])

model.load_weights(sys.argv[2])
model.save(sys.argv[2])
