import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import os
from timeit import default_timer as timer

model = load_model(os.path.abspath("../weights/YOLO_Face.h5"), compile=False)
image_path = os.path.abspath("C:/Users/taren/Desktop/face_test_backup/CC4B73632D38_2021_0209_151904_905_0043.jpg")


# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename) #TODO - can a second load be avoided?
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = K.expand_dims(image, 0)
    return image, width, height




input_w, input_h = 416, 416

# input_w, input_h = 640, 640
# load and prepare image
image, image_w, image_h = load_image_pixels(image_path, (input_w, input_h))

for x in range(50):
    start_time = timer()
    yhat = model.predict_on_batch(image)
    end_time = timer()
    print('*** Processing time: {:.2f}ms'.format((end_time -
                                                        start_time) * 1000))


# print([a.shape for a in yhat])