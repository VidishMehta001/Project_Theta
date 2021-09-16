# /right/image_mono
import os
import rclpy
from rclpy.node import Node
import json
from sensor_msgs.msg import Image
from std_msgs.msg import String
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(os.getcwd())
import sys
sys.path.append("install/theta_pkgs/lib/python3.8/site-packages/theta_pkgs")
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from cv_bridge import CvBridge
import torch
import cv2
import time
## Util libraries
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, check_img_size, is_ascii, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
            
class ModelImg(Node):

	def __init__(self):
		super().__init__('model_img')
		#fileroot = '~/ros2_ws/src/theta_pkgs/theta_pkgs/'
		fileroot=""
		model_path = fileroot+"saved_model/"
		object_file_path = fileroot+'saved_model/object-detection.pbtxt'
		self.subscription = self.create_subscription(
			Image,
			'/color/image',
			self.model_callback,
			10)
		self.subscription  # prevent unused variable warning
		self.publisher_ = self.create_publisher(Image, '/right/image_mono2', 10)
		self.publisher2_ = self.create_publisher(String, '/centroid', 10)
		self.mdl_inf = ModelInference(weights="models/best.pt", confidence_threshold=0.3, iou_threshold=0.45, line_thickness=3)

	def model_callback(self, msg):
		
		# Model Function returns dict of items:location(x,y) + image with BB << Focus today
		msg, centroid_msg = self.mdl_inf.run_inference_single_image(msg)
		
		# publish Image with BB << Focus today
		msg2 = String()
		#Service call items:items:location(x,y,z) to local "DB" TODO
		msg2.data = json.dumps(centroid_msg, cls=NpEncoder)
		#self.get_logger().info('Got Image')
		self.publisher_.publish(msg)
		self.publisher2_.publish(msg2)

class ModelInference(object):
    
    def __init__(self, weights, confidence_threshold, iou_threshold, line_thickness):
        self.weights = weights
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.line_thickness = line_thickness
        self.device=select_device("") # initialise cuda device
        self.model, self.classes = self.load_weights()
        self.stride = self.set_stride()
        self.half = False # Use FP16 half-precision inference
        self.vizualise = False
        self.agnostic_nms = False # Class agnostic NMS
        self.hide_labels = False # for hiding labels
        self.image_size = [640,640]
        self.max_detections=1000
        self.hide_conf = False
        self.bridge = CvBridge()
        
    def load_weights(self):
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        return model, names
    
    def set_stride(self):
        return self.model.stride.max()
    
    def run_inference_single_image(self, image):
        image=self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        if len(image.shape)==2:
            img2 = cv.merge((image,image,image))
            image = img2
        cv2.imwrite("image.jpg", image)
        dataset = LoadImages("image.jpg", img_size=self.image_size, stride=self.stride, auto=False)
        self.image_size = check_img_size(self.image_size, s=self.stride)  # check image size
        ascii = is_ascii(self.classes) 
        self.model(torch.zeros(1, 3, *self.image_size).to(self.device).type_as(next(self.model.parameters())))  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            pred = self.model(img, augment=False, visualize=self.vizualise)[0]
            ascii = is_ascii(self.classes)
            # Apply non-maximal suppression
            pred = non_max_suppression(pred, self.confidence_threshold, self.iou_threshold, None, self.agnostic_nms, max_det=self.max_detections)
            
            ## Second-stage classifier (optional) - for two stage object detections
#            if classify:
#                pred = apply_classifier(pred, modelc, img, im0s)
            
            for i, det in enumerate(pred):
                s, im0, frames = '', im0s.copy(), getattr(dataset, 'frame', 0)
                s += '%gx%g ' % img.shape[2:]  # print string
                annotator = Annotator(im0, line_width=self.line_thickness, pil=not ascii)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.classes[int(c)]}{'s' * (n > 1)}, "  # add to string
    
                    # Write results
                    bboxes = []
                    detections =[]
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.classes[c] if self.hide_conf else f'{self.classes[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        bboxes.append([int(xyxy[1].cpu().detach().numpy()), int(xyxy[0].cpu().detach().numpy()), int(xyxy[3].cpu().detach().numpy()), int(xyxy[2].cpu().detach().numpy())])
                        detections.append(c)
                        
            centroid_dict = None
            if len(det) !=0:      
                centroid_dict = ModelInference.get_image_centroid(bboxes, detections)
            # Stream results
            im0 = annotator.result()
            im0 = bridge.cv2_to_imgmsg(im0, encoding="rgb8")
            
        return im0, centroid_dict

    
    
    @staticmethod
    def get_image_centroid(bboxes, classes):
        # Get the centroid of every bbox in relative coordinates
        # Bounding boxes are encoded as [y_min, x_min, y_max, x_max]
        centroid_dict = dict()
        
        for each_class in set(classes):
            bbox_nest = [bboxes[x] for x in [i for i,j in enumerate(classes) if j==each_class]]
            bbox_list = list()
            for bbox in bbox_nest:     
                y_pt = int((bbox[2]-bbox[0])/2+bbox[0])
                x_pt = int((bbox[3]-bbox[1])/2+bbox[1])
                bbox_list.append([x_pt,y_pt])
            
            centroid_dict[int(each_class)] = bbox_list
            
        return centroid_dict
				   
def main(args=None):
	rclpy.init(args=args)

	model_img = ModelImg()

	rclpy.spin(model_img)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	minimal_subscriber.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	print(os.getcwd())
	main()

