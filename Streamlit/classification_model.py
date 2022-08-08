import PIL
print('Pillow Version:', PIL.__version__)

from PIL import Image

from matplotlib import image
from matplotlib import pyplot
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
%matplotlib inline

#generate dataset
IMG_WIDTH=200
IMG_HEIGHT=200
img_path=r'/home/kamanda\Downloads\training_set\training_set'

def generate_dataset(img_path):
   
    img_data_array=[]
    class_name=[]
    
    for dir1 in os.listdir(img_path):
        print("Collecting images for: ",dir1)
        for file in os.listdir(os.path.join(img_path, dir1)):
       
            image_path = os.path.join(img_path, dir1,  file)
            image = cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            try:
                image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            except:
                break
            image = np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name
# extract the image array and class name
img_data, class_name =generate_dataset(img_path)

'''Convert to array'''
img_data=np.array(img_data)
class_name=np.array(class_name)
img_data.shape
'''Dog mapping'''
def dog_cat_mapping(a):
    if a=="dogs":
        return 1
    else:return 0
class_name=list(map(dog_cat_mapping,class_name))
class_name=np.array(class_name)
#input shape for model parameters
input_shape=img_data.shape[1:]
'''Modeling'''
from tensorflow.keras.applications import InceptionResNetV2
conv_base=InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(200,200,3))
#Define the model
def model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
    model=Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    return model


model = model()
#check structure
#model.summary()
conv_base.trainable = False
#compile the model
model.compile(optimizer = 'adam', 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
model.fit(x = img_data,y = class_name, epochs = 10)
'''Evaluate performance'''
IMG_WIDTH = 200
IMG_HEIGHT = 200
img_path =  r'/home/kamanda\Downloads\test_set\test_set'


# extract the image array and class name
img_data_test, class_name_test = generate_dataset(r'/home/kamanda\Downloads\test_set\test_set')

img_data_test = np.array(img_data_test)
class_name_test = list(map(dog_cat_mapping,class_name_test))
class_name_test = np.array(class_name_test)
#make predictions
preds = model.predict(img_data_test).round().astype(int)
flat_pred = [item for sublist in preds for item in sublist]
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(class_name_test, flat_pred)

print("The Accuracy is: %2f" % accuracy)

#incorrect predictions
incorrects = np.nonzero(model.predict(img_data_test).round().astype(int).reshape((-1,)) != class_name_test)

#Save the model
from keras.models import load_model

model.save('catdog.h5')  # generates a HDF5 file 'your_model.h5'
from keras.models import save_model
#model = save_model(model,'/cat_dog.h5')
IMG_WIDTH=20
IMG_HEIGHT=200
#img_path='/home/kamanda\Downloads\test_set\test_set'
model.save('./models_catdog', save_format='tf')
