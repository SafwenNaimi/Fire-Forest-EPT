from pyimagesearch.gradcam import GradCAM
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16,Xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2


# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to the input image")
#ap.add_argument("-m", "--model", type=str, default="vgg",
#	choices=("vgg", "resnet", "covid19"),
#	help="model to be used")
#args = vars(ap.parse_args())

# initialize the model to be VGG16
Model = Xception

labels=  {'0': 0, '1': 1}
class_labels=['0 - No Fire', '1 - Fire']
# check to see if we are using ResNet
#if args["model"] == "resnet":
#	Model = ResNet50

# check to see if we are using ResNet
#elif args["model"] == "covid19":
#	Model = ('/home/ag/keras-covid-19/covid19ct2.model')
#def subtract_median_bg_image(im):
    #k = np.max(im.shape)//20*2+1
    #bg = cv2.medianBlur(im, 13)

    #print(k)
    #return cv2.addWeighted (im, 4, bg, -4, 128)



new_model = load_model("model5/model.h5")
from tensorflow.keras.preprocessing import image
# Check its architecture
new_model.summary()
path="testing/dilly_fire_1.jpg"
image_path = path

test_img_load = image.load_img(image_path, target_size=(128,128,3))

#test_img_load=subtract_median_bg_image(test_img_load)
test_img = image.img_to_array(test_img_load)
test_img = np.expand_dims(test_img, axis=0)
test_img /= 255

label_map_inv = {v:k  for k,v in labels.items()}

result = new_model.predict(test_img)
print(result)

prediction = result.argmax(axis=1)
print("pred",prediction)

i = label_map_inv[int(prediction)]
label=class_labels[(int(i))]
print(label)
print("res",result[0][int(i)])

# load the pre-trained CNN from disk
print("[INFO] loading model...")
#model = Model(weights="imagenet")
#model = load_model("BEST_MODEL_FINAL.h5", custom_objects=None, compile=True)

# load the original image from disk (in OpenCV format) and then
# resize the image to its target dimensions

# decode the ImageNet predictions to obtain the human-readable label
#decoded = imagenet_utils.decode_predictions(preds)
#(imagenetID, label, prob) = decoded[0][0]
#label = "{}: {:.2f}%".format(label, prob * 100)
#print("[INFO] {}".format(label))

image = load_img(path, target_size=(128, 128))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)

orig = cv2.imread(path)
resized = cv2.resize(orig, (128, 128))

# initialize our gradient class activation map and build the heatmap
cam = GradCAM(new_model, int(i))
heatmap = cam.compute_heatmap(test_img)

# resize the resulting heatmap to the original input image dimensions
# and then overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.3)

# draw the predicted label on the output image
#cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
#cv2.putText(output, "re", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
# display the original image and resulting heatmap and output image
# to our screen
output = np.hstack([orig, output])
output = imutils.resize(output, height=500)
print(orig.shape)
cv2.putText(output, label, (10, 40), 1, 2, (0, 255, 0), 2)
#cv2.putText(output, str(prediction), (10, 80), 1, 2, (0, 255, 0), 2)
cv2.imshow("Output", output)
cv2.imwrite("test_0.png",output)
cv2.waitKey(0)
