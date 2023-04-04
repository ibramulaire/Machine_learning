import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import array_to_img, img_to_array, load_img
import matplotlib.image as mpimg
import os



def load_images(folder):
    images = []
    imageslabel=[]
    for filename1 in os.listdir("../../DATA/"+folder):
        for filename in os.listdir("../../DATA/"+folder+"/"+filename1):
            img = mpimg.imread(os.path.join("../../DATA/"+folder+"/"+filename1, filename))
            if img is not None:
                images.append(img)
                imageslabel.append(filename1)
    return np.asarray(images),pd.DataFrame(imageslabel)

def testConvcouche(nbf1,nbf2,nbf3,taillekernal):
    activations = ['linear', 'relu',  'tanh', 'elu', 'selu']
    model = []
    param=[]
    
    for k in range(8):
        for i in range (len(activations)):
            activation=activations[i]
            for j in range(3):
                m = Sequential()
                m.add(Conv2D(nbf1,kernel_size=taillekernal,padding='same',activation=activation,
                        input_shape=(28,28,1)))
                m.add(MaxPool2D())
                if j>0:
                    m.add(Conv2D(nbf2,kernel_size=taillekernal,padding='same',activation=activation))
                    m.add(MaxPool2D())
                if j>1:
                    m.add(Conv2D(nbf3,kernel_size=taillekernal,padding='same',activation=activation))
                    m.add(MaxPool2D(padding='same'))
                m.add(Flatten())
                if k>0:
                    m.add(Dense(2**(k+4), activation=activation))
                
                m.add(Dropout(0.5))
                m.add(Dense(10, activation='softmax'))
                m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                model.append(m)
                param.append("model_nbconv="+str(j)+"_act="+str(activation)+"_kernal zize="+str(taillekernal)+"dense:="+str(2**(k+4))+"_filt1_2_3="+str(nbf1)+"_"+str(nbf2)+"_"+str(nbf3))
    return model ,param
folder="brut"
num_classes = 10
input_shape = (28, 28, 1)
x_train, y_train=load_images(folder+"/train")
x_val, y_val=load_images(folder+"/val")
x_test, y_test=load_images(folder+"/test")

y_test=y_test[0]
y_train=y_train[0]
y_val=y_val[0]

y_test=np.asarray(y_test.apply(lambda x : ord(x)-ord('A')))
y_train=np.asarray(y_train.apply(lambda x : ord(x)-ord('A')))
y_val=np.asarray(y_val.apply(lambda x : ord(x)-ord('A')))


x_train = x_train.astype("float32")
x_test = x_test.astype("float32") 
x_val = x_val.astype("float32")

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
x_val = np.expand_dims(x_val, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
print(x_val.shape[0], "val samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

img_width, img_height = 28, 28
top_model_weights_path = 'poidsNotMnist_A2jdata'+folder+'.h5'
train_data_dir = "../../DATA/"+folder+'/train/'
validation_data_dir = "../../DATA/"+folder+'/val/'
test_data_dir="../../DATA/"+folder+'/test/'
nb_train_samples = x_train.shape[0],
nb_validation_samples = x_val.shape[0]
nb_test_samples=x_test.shape[0]

def execute(folder,epochs,batch_size,nbf1,nbf2,nbf3,taillekernal):
   

    model,param=testConvcouche(nbf1,nbf2,nbf3,taillekernal)
    resultatfit=[]
    scoretest=[]
    print(str(len(model)))
    for i in range (len(model)):
        print(param[i])
        result=model[i].fit(x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val))
        score = model[i].evaluate(x_test,y_test, verbose=0)
        resultatfit.append(result)
        scoretest.append(score)
    for i in range (len(model)):
        scorede=pd.DataFrame(scoretest[i])
        scorede.to_csv(param[i]+"_e="+str(epochs)+"_b="+str(batch_size)+"test.csv", index=False)
        history_df = pd.DataFrame(resultatfit[i].history)
        history_df.to_csv(param[i]+"_e="+str(epochs)+"_b="+str(batch_size)+"entrainement.csv", index=False)
        model[i].save(param[i]+"_e="+str(epochs)+"_b="+str(batch_size)+'model.h5')


execute("brut",20,16,24,48,64,5)
execute("brut",20,32,24,48,64,5)
execute("brut",20,64,24,48,64,5)

