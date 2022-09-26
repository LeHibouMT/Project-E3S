#Code reprit depuis https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
#Pour les questions, voir la page ou demander au chef de projet Michel Taing

import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import os.path
datapath = r"C:\Users"
datapath = os.path.join(datapath,"Michel","Desktop","PROJET","BDD","bases")

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory=datapath+r"\base_sp",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory=datapath+r"\base_test", target_size=(224,224))

# C'est un réseau de neurone convolutif et donc profond, l'architecture choisi est le VGG-16 
# un transfer learning ne devrait pas être nécessaire si nous avons une bonne base de données à disposition
# écrire cela avec des boucles peut aussi être plus lisible
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4, activation="softmax"))

from tensorflow.keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# pour avoir un résumé du modèle
#model.summary()

from keras.models import load_model
model = load_model(datapath+r"\checkpoint"+r"\vgg16_1.h5")

#from keras.callbacks import ModelCheckpoint, EarlyStopping
#checkpoint = ModelCheckpoint(datapath+r"\checkpoint"+r"\vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
#hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=5,callbacks=[checkpoint,early])
#model.save(datapath+r"\modelvgg.h5")

# pour visualiser l'entraînement et vérifier que nous avons des résultats satisfaisants, éviter l'overtraining et l'overfitting 
#import matplotlib.pyplot as plt
#plt.plot(hist.history['accuracy'])
#plt.plot(hist.history['val_accuracy'])
#plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])
#plt.title("model accuracy")
#plt.ylabel("Accuracy")
#plt.xlabel("Epoch")
#plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
#plt.show()


datapath2 = r"C:\Users"
datapath2 = os.path.join(datapath2,"Michel","Desktop","basetraincv")
datapathbad = os.path.join(datapath2,"bad")
datapathgood = os.path.join(datapath2,"good")
datapathgreat = os.path.join(datapath2,"great")
datapathpassable = os.path.join(datapath2,"passable")
compteur = 0
result = 0
for file in os.listdir(datapathbad) :
    img = image.load_img(os.path.join(datapathbad,file),target_size=(224,224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    output = model.predict(img)
    if output[0][0]>output[0][1] and output[0][0]>output[0][2] and output[0][0]>output[0][3] :
        result+=1
    compteur+=1
print("résultat prédiction bad")
print(result/compteur)

compteur = 0
result = 0
for file in os.listdir(datapathgood) :
    img = image.load_img(os.path.join(datapathgood,file),target_size=(224,224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    output = model.predict(img)
    if output[0][1]>output[0][0] and output[0][1]>output[0][2] and output[0][1]>output[0][3] :
        result+=1
    compteur+=1
print("résultat prédiction good")
print(result/compteur)

compteur = 0
result = 0
for file in os.listdir(datapathgreat) :
    img = image.load_img(os.path.join(datapathgreat,file),target_size=(224,224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    output = model.predict(img)
    if output[0][2]>output[0][0] and output[0][2]>output[0][1] and output[0][2]>output[0][3] :
        result+=1
    compteur+=1
print("résultat prédiction great")
print(result/compteur)

compteur = 0
result = 0
for file in os.listdir(datapathpassable) :
    img = image.load_img(os.path.join(datapathpassable,file),target_size=(224,224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    output = model.predict(img)
    if output[0][3]>output[0][0] and output[0][3]>output[0][2] and output[0][3]>output[0][1] :
        result+=1
    compteur+=1
print("résultat prédiction passable")
print(result/compteur)
