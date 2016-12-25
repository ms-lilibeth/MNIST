import sys
import os
from keras.models import model_from_json
import cv2

img_rows, img_cols = 28, 28  # input image dimensions


filename = "mnist"
print("Loading model...")
model = None
if os.path.exists(filename + '.arch.json'):
    with open(filename + '.arch.json', 'r') as f:
        model = f.read()
        model = model_from_json(model)

if model is None:
    print("Couldn't read the model")
    exit()
print("Loading weights...")
if os.path.exists(filename + '.weights.h5'):
    print('Loading weights...')
    model.load_weights(filename + '.weights.h5')
else:
    print("Couldn't load the weights")
    exit()

while True:
    inp = input("Filepath or exit: ")
    if inp == "exit":
        break
    if not os.path.exists(inp):
        print("Path does not exist")
        continue
    try:
        img = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)
    except:
        print("Could not read the image")
    img = cv2.resize(img, (img_rows, img_cols))
    img = img.reshape((1, 1, img_rows, img_cols))
    result = model.predict(img)
    print("Result: ", result[0].tolist().index(1.))
# mnist_img/mnist0.png