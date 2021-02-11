from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import datetime



class_labels=['0 - No Fire', '1 - Fire']

model = load_model("model5/model.h5")

start = datetime.datetime.now()


orig = cv2.imread("testing/48228796_403.jpg")

#image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
image = cv2.resize(orig, (128, 128))
image = image.astype("float") / 255.0

image = img_to_array(image)
image = np.reshape(image,[1,128,128,3])
#image = np.expand_dims(image, axis=0)
# make predictions on the input image
pred = model.predict(image)
print(pred)
h=np.max(pred)
prob="{:.2f} %".format(h*100)
pred = pred.argmax(axis=1)[0]


end = datetime.datetime.now()

elapsed=end-start

res=class_labels[pred]


print(res)
orig = cv2.resize(orig, (700, 700))
cv2.putText(orig,"Category: "+ res,(10,20),1,1,(0,255,0),2)
cv2.putText(orig,"Latency: "+str(elapsed),(10,50),1,1,(0,255,0),2)
cv2.putText(orig,"Probability: "+str(prob),(10,80),1,1,(0,255,0),2)

#results.append(orig)

#montage = build_montages(results, (128, 128), (15, 15))[0]

cv2.imshow("Results", orig)
cv2.waitKey(0)

