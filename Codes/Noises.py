#Code reprit depuis https://pythonmana.com/2022/01/202201111035518790.html
#Pour les questions, voir la page ou demander au chef de projet Michel Taing
#Dataset https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
#Dataset https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

import os.path
import numpy as np
from PIL import Image
from os import listdir
import cv2
import random
datapath = r"C:\Users"
datapath = os.path.join(datapath,"Michel","Desktop","basetraincv")
databasepath = datapath+r"\base0"


def gaussien(image, mean=0, var=0.001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def sp(image, prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

#compteur = 0
#databasepathg = databasepath+"_gauss1"
#for file in os.listdir(databasepathg) :
#    image = cv2.imread(os.path.join(databasepathg,file))    #on ouvre l'image
#    imagetosave = gaussien(image,0.02) #on applique le bruit gaussien et on a en retour un np.array
#    imagetosave2 = gaussien(imagetosave,0.02) #on applique le bruit gaussien et on a en retour un np.array
#    imagetosave = Image.fromarray(imagetosave) #on transforme l'array en image
#    imagetosave2 = Image.fromarray(imagetosave2) #on transforme l'array en image
#    imagetosave.save(datapath+r"\full_archive_gauss2"+"/image"+str(compteur)+".jpeg","jpeg") #on enregistre l'image
#    imagetosave2.save(datapath+r"\full_archive_gauss3"+"/image"+str(compteur)+".jpeg","jpeg") #on enregistre l'image
#    compteur+=1

compteur = 0
for file in os.listdir(databasepath) :
    image = cv2.imread(os.path.join(databasepath,file))    #on ouvre l'image
    imagetosave1 = sp(image,0.02) #on applique le bruit sp et on a en retour un np.array
    imagetosave2 = sp(imagetosave1,0.02) #on applique le bruit sp et on a en retour un np.array
    imagetosave3 = sp(imagetosave2,0.02) #on applique le bruit sp et on a en retour un np.array
    imagetosave1 = Image.fromarray(imagetosave1) #on transforme l'array en image
    imagetosave2 = Image.fromarray(imagetosave2) #on transforme l'array en image
    imagetosave3 = Image.fromarray(imagetosave3) #on transforme l'array en image
    imagetosave1.save(datapath+r"\base1"+"/image"+str(compteur)+".jpeg","jpeg") #on enregistre l'image
    imagetosave2.save(datapath+r"\base2"+"/image"+str(compteur)+".jpeg","jpeg") #on enregistre l'image
    imagetosave3.save(datapath+r"\base3"+"/image"+str(compteur)+".jpeg","jpeg") #on enregistre l'image
    compteur+=1