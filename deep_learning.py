###import the modules

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


import tensorflow as tf
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.utils import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D ,Dropout, Flatten , MaxPooling2D
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))




### load the dataset

TRAIN_DIR='./train/train/'
TEST_DIR='./test/test/'

def load_dataset(directory):
    image_paths=[]
    labels=[]

    for label in os.listdir(directory):
        for filename in os.listdir(directory+label):
            image_path = os.path.join(directory,label,filename)
            image_paths.append(image_path)
            labels.append(label)
        #print(label,'completed')

    return image_paths,labels



### convert into dataframe

train = pd.DataFrame()
train['image'],train['label']= load_dataset(TRAIN_DIR)
# shuffle the dataset
train = train.sample(frac=1).reset_index(drop=True)
#print(train.head())

test = pd.DataFrame()
test['image'],test['label']= load_dataset(TEST_DIR)
#print(test.head())



### extracting features

def extract_features(images):
    features=[]
    for image in tqdm(images):
        img= load_img(image, color_mode='grayscale')
        img=np.array(img)
        features.append(img)
    
    features=np.array(features)
    features = features.reshape(len(features), 48,48,1)

    return features

train_features = extract_features(train['image'])
test_features = extract_features(test['image'])

x_train = train_features/255.0
x_test = test_features/255.0

###converts label to integre

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

y_train= to_categorical(y_train , num_classes=7)
y_test= to_categorical(y_test , num_classes=7)

#print(y_train[0])


### config
input_shape=(48, 48, 1)
output_class = 7

### Model creation

model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))


model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))


model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layers
model.add(Dense(output_class,activation='softmax'))

model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])

### train the model 
history= model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, validation_data=(x_train, y_train))

import pickle
import numpy as np

# Save x_test and test
np.save("x_test.npy", x_test)
with open("test_labels.pkl", "wb") as f:
    pickle.dump(test, f)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Save the model
model.save("my_model.h5")



### plot the results

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('loss Graph')
plt.legend()


plt.show()