# /right/image_mono

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

import onnx
import warnings
from onnx_tf.backend import prepare
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from cv_bridge import CvBridge
import os

class ModelImg(Node):

	def __init__(self):
		super().__init__('model_img')
		self.subscription = self.create_subscription(
			Image,
			'/right/image_mono',
			self.model_callback,
			10)
		self.subscription  # prevent unused variable warning
		self.publisher_ = self.create_publisher(Image, '/right/image_mono2', 10)
		self.mdl_inf = ModelInf();

	def model_callback(self, msg):
		
		# Model Function returns dict of items:location(x,y) + image with BB << Focus today
		msg = self.mdl_inf.inf_run(msg)
		# Function take in items:location(x,y), returns items:location(x,y,z) TODO
		
		# publish Image with BB << Focus today
		
		#Service call items:items:location(x,y,z) to local "DB" TODO
		
		self.get_logger().info('Got Image')
		self.publisher_.publish(msg)

class ModelInf:
	def __init__(self):
		warnings.filterwarnings('ignore') # Ignore all the warning messages in this tutorial
		model = onnx.load('model/ssd-mobilenet.onnx') # Load the ONNX file
		self.tf_rep = prepare(model) # Import the ONNX model to Tensorflow
		self.bridge = CvBridge()
	def inf_run(self,image_message):
		img=self.bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')
		print(img.shape)
		if len(img.shape)==2:
			img2 = cv.merge((img,img,img))
			img = img2
		img = cv.resize(img,(300,300))
		img2 = np.array([np.rollaxis(img, 2, 0)  ])
		img2=img2.astype(np.float32)
		output = self.tf_rep.run(img2)
		for i in range(len(output[0][0])):
			curv = output[0][0][i]
			max_loc = np.where(curv==max(curv))[0][0]
			if max_loc > 0:
				#print(output[1][0][i])
				bv = output[1][0][i]
				st = (int(bv[1]*300),int(bv[0]*300))
				end = (int(bv[3]*300),int(bv[2]*300))
				img=cv.rectangle(img,st,end, (255,0,0), 2)
		image_message = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
		return image_message
				   
def main(args=None):
	print(os.getcwd())
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

