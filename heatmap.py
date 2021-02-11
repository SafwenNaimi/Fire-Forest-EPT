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





Model = Xception

labels=  {'0': 0, '1': 1}
class_labels=['0 - No Fire', '1 - Fire']




new_model = load_model("model.h5")
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




image = load_img(path, target_size=(128, 128))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)

orig = cv2.imread(path)
resized = cv2.resize(orig, (128, 128))

cam = GradCAM(new_model, int(i))
heatmap = cam.compute_heatmap(test_img)


heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.3)



output = np.hstack([orig, output])
output = imutils.resize(output, height=500)
print(orig.shape)
cv2.putText(output, label, (10, 40), 1, 2, (0, 255, 0), 2)
#cv2.putText(output, str(prediction), (10, 80), 1, 2, (0, 255, 0), 2)
cv2.imshow("Output", output)
cv2.imwrite("test_0.png",output)
cv2.waitKey(0)
