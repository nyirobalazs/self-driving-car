import os
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout,BatchNormalization,Activation
from tensorflow.keras.optimizers import Adam
from imgaug import augmenters
import random
from matplotlib import pyplot as plt

# MODEL
def define_model():
  model = Sequential(name='NVIDIA')
  
  model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(240,320,3), padding='same'))
  model.add(Activation('elu'))
  model.add(Dropout(0.2))
  
  model.add(Conv2D(36, (5, 5), strides=(2, 2)))
  model.add(Activation('elu'))
  model.add(Dropout(0.2))
  
  model.add(Conv2D(48, (5, 5), strides=(2, 2)))
  model.add(Activation('elu'))
  model.add(Dropout(0.2))
  
  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('elu'))
  model.add(Dropout(0.2))
  
  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('elu'))
  model.add(Dropout(0.2))

  # Fully Connected Layers
  model.add(Flatten())
  model.add(Dense(120))
  model.add(Activation('elu'))
  model.add(Dropout(0.5))
  model.add(Dense(50))
  model.add(Activation('elu'))
  model.add(Dropout(0.5))
  model.add(Dense(10))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  
  ### Output layer
  model.add(Dense(2,activation='sigmoid',name='output'))
  
  opt = Adam(learning_rate=3e-4)
  
  model.compile(loss='mse',optimizer=opt,
                metrics=['mse'])
  
  return model

### Data augmentation protocols

def zoom(image):
  zoom = augmenters.Affine(scale=(1,1.25))
  image = zoom.augment_image(image)
  return image

def adjust_brightness(image):
  brightness = augmenters.Multiply((0.7,1.3))
  image = brightness.augment_image(image)
  return image

def blur(image):
  kernel_size = random.randint(1,5)
  image = cv2.blur(image,(kernel_size,kernel_size))
  return image

def flip(image,angle):
  image = cv2.flip(image,1)
  angle = 1 - angle
  return image,angle

def random_augment(image,angle):
  if np.random.randint(0,1) == 1:
    image = zoom(image)
  if np.random.randint(0,1) == 1:
    image = adjust_brightness(image)
  if np.random.randint(0,1) == 1:
    image = blur(image)
  if np.random.randint(0,1) == 1:
    image = flip(image,angle)
  return image,angle


#yield given number of preprocessed images
def batch_generator(x,y,batch_size,training):
  while True:
    batchX = []
    batchY = []
    nsteps = int(len(x)/batch_size)
    chunk = list(range(nsteps))
    random.shuffle(chunk)
    i = chunk[0]
    batch = np.arange(i*batch_size,(i+1)*batch_size)
    for j in batch:
      image = cv2.imread(x[j])
      angle = y[j,0]
      if training == True:
        image,angle = random_augment(image,angle)
      batchX.append(image)
      batchY.append([angle,y[j,1]])
    chunk = chunk[1:]
    batchX = np.asarray(batchX)
    batchY = np.asarray(batchY)
    yield (batchX,batchY)
    
#Load data from directory
dataDir = '../data/training_data/' #change folder path
fileList = os.listdir(dataDir)
x = []
y = []
extension = '.png'

with open('../data/training_data/training_norm.csv',newline='') as csvfile:
  datapoints = csv.reader(csvfile,delimiter=',')
  next(datapoints) # Skip the first row
  for row in datapoints:
    filepath = dataDir + str(row[0]) + extension
    x.append(filepath)
    y.append([row[1],row[2]])
    
y = np.stack(y).astype(np.float32)

#split dataset into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(x, y,
                                      test_size=0.2, shuffle=True)
  
print('Compiling model...')
model = define_model()

print(model.summary())

#define training parameters
batch_size = 128
steps = int(len(X_train)/batch_size)
val_batch_size = 128
val_steps = int(len(X_valid)/val_batch_size)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  'v4_checkpoint.h5',
  monitor='val_loss',
  verbose=1,
  save_best_only=True,)
  
print('Training...')
history = model.fit(batch_generator(X_train,y_train,batch_size,True),
                              steps_per_epoch=steps,
                              epochs=200,
                              validation_data=batch_generator(
                                X_valid,y_valid,val_batch_size,False),
                              validation_steps=steps,
                              shuffle=1,
                              verbose=2,
                              callbacks=checkpoint_callback)
                              
model.save('v4_model.h5')
print('Model saved.')

#Create loss and validation loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.yscale('log')
plt.legend(['Training loss','Validation loss'])
plt.savefig('v4_training'+str(datetime.now())+'.png')

    
