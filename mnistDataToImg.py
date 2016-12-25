from six.moves import cPickle
from cv2 import imwrite
dataset_path = "mnist.pkl"
data = None

with open(dataset_path, 'rb') as f:
    data = cPickle.load(f, encoding="bytes")
if data is None:
    print("Couldn't load MNIST data")
    exit()
(X_train, y_train), (X_test, y_test) = data
for i in range(100):
    filename = "mnist_img/mnist" + str(i) + ".png"
    imwrite(filename, X_test[i])