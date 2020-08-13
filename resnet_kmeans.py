# RUN python resnet_kmeans.py --input_dir /home/shahmustafa/Desktop/dataset/ --target_dir ./outdir_sub/"
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from PIL import Image as pil_image

# # import for using GPU
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

image.LOAD_TRUNCATED_IMAGES = True
model = ResNet50(weights='imagenet', include_top=False)

# constructing the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True, help="path to input dataset dir")
ap.add_argument("-t", "--target_dir", required=True, help="target dir")
args = vars(ap.parse_args())

# Variables
imdir = args["input_dir"]
targetdir = args["target_dir"]

number_clusters = 6

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.jpg'))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    img = image.load_img(imagepath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())

# Clustering
kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(featurelist))

# Copy images renamed by cluster
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_):
    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
    sub_dir = str(m)
    try:
        os.makedirs(targetdir + sub_dir)
    except OSError:
        pass
    shutil.copy(filelist[i], targetdir + sub_dir + '/' + str(m) + "_" + str(i) + ".jpg")
