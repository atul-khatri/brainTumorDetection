import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

INPUT_SIZE = 64

modl = load_model("brain_tumor_model.h5")

image = cv2.imread("datasets/pred/pred5.jpg")
img = Image.fromarray(image)
img = img.resize((INPUT_SIZE, INPUT_SIZE))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)
result = modl.predict(input_img)
print(result)
