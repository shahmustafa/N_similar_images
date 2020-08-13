# RUN python classification.py --input_img /data1/prjs/OCR/ocrmypdf/outdir_sub/5/5_52.jpg --N_req_imgs 4 --load_w /home/shahmustafa/Desktop/bottleneck_fc_model_save.h5
# import the necessary packages
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import numpy as np
import matplotlib.pyplot as plt
import argparse

# # import for using GPU
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# constructing the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_img", required=True, help="path to input image")
ap.add_argument("-N", "--N_req_imgs", required=True, help="N request Image/s similar to input image")
ap.add_argument("-N", "--load_w", required=True, help="load trainde weights")
args = vars(ap.parse_args())

num_classes = 6

# Target image
path = args["input_img"]
# Number of requested images
N = args["N_req_imgs"]


def read_image(file_path, resize=True):
    image_path = file_path

    # orig = mpimg.imread(image_path)

    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(512, 512))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    if resize:
        image = image / 255

    image = np.expand_dims(image, axis=0)
    return image

image_pred = read_image(path)

# build the VGG16 network
vgg16 = applications.VGG16(include_top=False, weights='imagenet')

# get the bottleneck prediction from the pre-trained VGG16 model
bottleneck_prediction = vgg16.predict(image_pred)

# Rebuilds top model using weights....
# build top model
model = Sequential()
model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.load_weights(args["load_w"])

# use the bottleneck prediction on the top model to get the final classification
class_predicted = model.predict_classes(bottleneck_prediction)
label = int(class_predicted)

print("Label: {}".format(label))

cat_dir = '/data1/prjs/OCR/ocrmypdf/outdir_sub/'

class_dir = cat_dir + str(label)
# print(class_dir)


from os import listdir
from os.path import isfile, join
import numpy
import cv2
from PIL import Image

mypath = class_dir
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images = numpy.empty(len(onlyfiles), dtype=object)
# print(images.size)
# dim = numpy.empty(len(onlyfiles), dtype=object)
total_width = 0
total_hight = 0
dim = 0

for n in range(0, len(onlyfiles)):
    # print(n)
    if n == N:
        break
    images[n] = cv2.imread(join(mypath, onlyfiles[n]))
    # im = cv2.imread( join(mypath,onlyfiles[n]) )
    # cv2.imshow('dd',im)
    img = numpy.array(images[n])
    dim = img.shape
    width = dim[0]
    hight = dim[1]
    # print(width)
    total_width += width
    total_hight += hight

#print(total_hight, total_width)

new_im = Image.new('RGB', (total_width, total_hight))
# print(new_im.size)
x_offset = 0
y_offset = 0
maximum = 0
pos = (5, 5)
img = []
for n in range(0, len(onlyfiles)):
    if n == N:
        break
    # print(onlyfiles[n])
    img1 = Image.open(join(mypath, onlyfiles[n]))
    # print(img1)
    # if x_offset <= total_hight :
    new_im.paste(img1, (x_offset, y_offset))
    # if y_offset <= total_hight or x_offset <= total_width :
    x_offset = 20 + x_offset + img1.size[0]

    next_value = img1.size[1]
    if maximum <= next_value:
        maximum = next_value

    if x_offset >= 1600:
        x_offset = 0
        y_offset = 20 + maximum + y_offset
        maximum = 0

new_im.save('Similar_images.jpg')

# displaying the input image
plt.title("Input image")
in_im = cv2.imread(path)
plt.imshow(in_im)
plt.show()

# displaying the N requested images
plt.title("N={} Resquested Image/s similar to input image".format(N))
plt.imshow(new_im)
plt.show()
# cv2.imshow('image',new_im)
