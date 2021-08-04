# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 23:08:14 2021

@author: vidis
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 21:45:19 2020

@author: vidis
"""

import numpy as np
import sys
import tensorflow as tf
import cv2
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from PIL import Image


class ModelInference (object):
    
    def __init__(self, model_path, object_file_path):
        self.model = tf.saved_model.load(model_path)
        self.num_classes = 1
        self.label_map = label_map_util.load_labelmap(object_file_path)
        categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes = self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
    
    @staticmethod
    def run_inference_for_single_image(model, image):
          image = np.asarray(image)
          # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
          input_tensor = tf.convert_to_tensor(image)
          # The model expects a batch of images, so add an axis with `tf.newaxis`.
          input_tensor = input_tensor[tf.newaxis,...]
        
          # Run inference
          model_fn = model.signatures['serving_default']
          output_dict = model_fn(input_tensor)
        
          # All outputs are batches tensors.
          num_detections = int(output_dict.pop('num_detections'))
          output_dict = {key:value[0, :num_detections].numpy() 
                         for key,value in output_dict.items()}
          output_dict['num_detections'] = num_detections
        
          # detection_classes should be ints.
          output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
           
          return output_dict
      
     
    @staticmethod
    def show_inference_single_image(model, image_path, category_index, image_np=""):
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      if image_np == "":
          image_np = np.array(Image.open(image_path))
      # Actual detection.
      output_dict = ModelInference.run_inference_for_single_image(model, image_np)
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks_reframed', None),
          use_normalized_coordinates=True,
          line_thickness=8)
    
      if image_path != "":
          cv2.imshow("window", cv2.resize(image_np, (750,800)))
          cv2.waitKey(0)
          cv2.destroyAllWindows()
    
      return image_np
  
    def inference_video(self, video_path):
        
        # loaded videos - for camera please change here
        cap = cv2.VideoCapture(video_path)
        
        # main loop
        while(cap.isOpened()):
            ret, image = cap.read()
            if ret:
                image = ModelInference.show_inference_single_image(self.model, image_path="", category_index=self.category_index, image_np=image)
                # Display the resulting image
                cv2.imshow('image', image)
               
                if cv2.waitKey(25) & 0xFF == ord('q'):
                  break
              
        # the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()

model_path = "saved_model/"
object_file_path = 'object-detection.pbtxt'

# Testing on the image - test.jpg
image_file_path = "Test1.jpg"

# Testing on single image
MI = ModelInference(model_path, object_file_path)
MI.show_inference_single_image(MI.model, image_file_path,MI.category_index)


# video_path = ""

# # Testing on video
# MI = ModelInference(model_path, object_file_path)
# MI.inference_video(video_path)