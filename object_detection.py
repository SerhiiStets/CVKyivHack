
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image

import cv2

if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')


# ## Env setup


# This is needed to display the images.
# get_ipython().run_line_magic('matplotlib', 'inline')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

reload_model = False
if reload_model:
  opener = urllib.request.URLopener()
  opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
  tar_file = tarfile.open(MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code

# def load_image_into_numpy_array(image):
#   (im_width, im_height) = image.size
#   return np.array(image.getdata()).reshape(
#       (im_height, im_width, 3)).astype(np.uint8)


class CarDetector:
  def __init__(self):
    with detection_graph.as_default():
      self.sess = tf.Session(graph=detection_graph)
      # Definite input and output Tensors for detection_graph
      self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
      self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
      self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

  def getBoxes(self, image, width, height):
    image_np = image
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = self.sess.run(
        [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
        feed_dict={self.image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
    new_boxes = []
    new_classes = []
    new_scores = []

    for i in range(len(boxes[0])):
      box = boxes[0][i]
      if box[1] > 0.5:
        if box[3]-box[1] > 0.1 and scores[0][i] > 0.4 and (
            int(classes[0][i]) == 3 or int(classes[0][i]) == 6 or int(classes[0][i]) == 8):
          new_boxes.append(box)
          new_classes.append(classes[0][i])
          new_scores.append(scores[0][i])

    # print(categories)
    # for item in label_map.item:
    #   print("{}:{}".format(label_map.id, label_map.display_name))

    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     np.squeeze(new_boxes),
    #     np.squeeze(new_classes).astype(np.int32),
    #     np.squeeze(new_scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=8)

    labels = {3: "Car", 6: "Bus", 8: "Truck"}
    # print(new_boxes)
    for i in range(len(new_boxes)):
      ly, lx, hy, hx = new_boxes[i]
      cx = (hx+lx)/2
      cy = (hy+ly)/2
      cv2.putText(image_np, "{}".format(labels[new_classes[i]]),
                  (int(cx*width-50), int(cy*height-30)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
      cv2.rectangle(image_np, (int(lx*width), int(ly*height)),
                              (int(hx*width), int(hy*height)), (0, 200, 0), 5)

    return (new_boxes, image_np)

  def __del__(self):
    pass
