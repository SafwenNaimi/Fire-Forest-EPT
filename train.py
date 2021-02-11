import datetime
import os, sys, shutil
import numpy as np
import cv2
from numpy import loadtxt
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix, cohen_kappa_score,classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers, applications
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import TensorBoard
import time




train_dir = 'fire_dataset/train/'
validation_dir = 'fire_dataset/test/'


bacth_size = 8
epochs = 150
warmup_epocks = 2
learning_rate = 0.000001 #0.00001
warmup_learning_rate = 0.00008
height = 128
width = 128
colors = 3
n_classes = 2
es_patience = 18
rlrop_patience = 3
decay_drop = 0.5
based_model_last_block_layer_number = 0#100




train_datagen = ImageDataGenerator(
      rescale=1/255,
      rotation_range=10,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.5,
      brightness_range=[0.7,1.3],
      horizontal_flip=True,
      fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(height, width),
        batch_size= bacth_size,
        shuffle = True,
        class_mode= 'categorical')

val_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(height, width),
        batch_size = bacth_size,
        shuffle=True,
        class_mode= 'categorical')



import efficientnet.tfkeras as eft
from keras.layers import Flatten,GlobalMaxPooling2D
from keras import regularizers
from tensorflow.keras.layers import BatchNormalization

def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = applications.ResNet50(weights='imagenet',  # à modifier
                                        include_top=False,
                                        input_tensor=input_tensor)
    print(base_model.summary())
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(512, activation='relu', name='output')(x)  # 1024
    output = Dropout(0.5)(output)

    model_prim = Model(input_tensor, output)
    final_output = Dense(n_out, activation='softmax', name='final_output')(model_prim.output)
    model = Model(input_tensor, final_output)

    return model

model = create_model(input_shape=(height, width, colors), n_out=n_classes)





from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import glorot_uniform



for layer in model.layers[:based_model_last_block_layer_number]:
    layer.trainable = False
for layer in model.layers[based_model_last_block_layer_number:]:
    layer.trainable = True







from tensorflow.keras import metrics

metric_list=['accuracy']
optimizer = optimizers.Adam(lr=warmup_learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)





rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=rlrop_patience, factor=decay_drop, min_lr=1e-6, verbose=1)


from tensorflow.keras import metrics
metric_list=['accuracy']
optimizer = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss="binary_crossentropy",  metrics=metric_list)


print(model.summary())



step_train = train_generator.n//train_generator.batch_size
step_validation = val_generator.n//val_generator.batch_size

print(train_generator.n)

checkpointer = ModelCheckpoint(filepath='model.h5',monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')


callback_list = [rlrop,checkpointer]


history_warmup = model.fit_generator(generator=train_generator,
                              steps_per_epoch=step_train,
                              validation_data=val_generator,
                              validation_steps=step_validation,
                              callbacks=[checkpointer],
                              epochs=1,
                              verbose=1).history



history = model.fit_generator(generator=train_generator,
                             steps_per_epoch=step_train,
                             validation_data=val_generator,
                              validation_steps=step_validation,
                              epochs=epochs,
                              callbacks=callback_list,
                              verbose=1).history



model.save('modell.h5')


train_data = pd.read_csv('fire_dataset/train_2.csv')
test_data = pd.read_csv('fire_dataset/test_2.csv')

train_path = 'fire_dataset/all/train/'
test_path = 'fire_dataset/all/test/'
import seaborn as sns



def preprocess_image(image_path, desired_size=128):
    im=cv2.imread(image_path)
    im=cv2.resize(im,(desired_size,) * 2)
    #im = Image.open(image_path)
    #im = im.resize((desired_size,) * 2, resample=Image.LANCZOS)
    return im


N = train_data.shape[0]

x_train = np.empty((N, 128, 128, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(train_data['id_code'])):

    x_train[i, :, :, :] = preprocess_image(
        os.path.join(train_path + image_id )
    )

x_train = x_train / 255
# use the model to generate predictions for all of the training images
start = datetime.datetime.now()
print('Started predicting at {}'.format(start))

train_prediction = model.predict([x_train])

end = datetime.datetime.now()
elapsed = end - start
print('Predicting took a total of {}'.format(elapsed))

# take the highest predicted probability for each image
train_predictions = [np.argmax(pred) for pred in train_prediction]



# look at how the model performed for each class
labels = ['0 - No Fire', '1 - Fire']
cnf_matrix = confusion_matrix(train_data['diagnosis'].astype('int'), train_predictions)
cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cnf_matrix_norm, index=labels, columns=labels)
plt.figure(figsize=(16, 7))
sns.heatmap(df_cm, annot=True, fmt='.2f', cmap="Blues")
plt.show()


print(classification_report(train_data['diagnosis'].astype('int'), train_predictions, target_names=labels))


import matplotlib.pyplot as plt

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("result.png")




