# /right/image_mono
import os
import rclpy
from rclpy.node import Node
import json
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge
import numpy as np


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


class CoordTransfer(Node):

	def __init__(self):
		super().__init__('coord_transfer')
		self.subscription = self.create_subscription(
			String,
			'/centroid',
			self.model_callback,
			10)
		self.subscription  # prevent unused variable warning
		self.publisher_ = self.create_publisher(String, '/centroid_depth', 10)
		
		self.subscription2 = self.create_subscription(
			Image,
			'/stereo/depth',
			self.model_callback2,
			10)
		self.subscription2  # prevent unused variable warning
		self.image_data  = []
		self.bridge = CvBridge()
	def model_callback(self, msg):
		data = json.loads(msg.data)
		
		if len(self.image_data)==0 or len(data)==0:
			return
		image_shape = (self.image_data.shape[0]-1,self.image_data.shape[1]-1)
		
		new_coords =[]

		for item in data:
			location = [int(image_shape[0]*item[0]),int(image_shape[1]*item[1])]
			z_val = self.image_data[location[0],location[1]][0]
			location.append(z_val)

			new_coords.append(location)

		new_msg = String()
		print("<<<<<<<<<<<<")
		print(new_coords)
		new_msg.data = json.dumps(new_coords, cls=NpEncoder)
		self.get_logger().info('Got coords...')
		self.publisher_.publish(new_msg)
		
	def model_callback2(self, msg):
		image_np=self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
		if len(image_np.shape)==2:
			img2 = cv.merge((image_np,image_np,image_np))
			image_np = img2
		self.image_data = image_np

def main(args=None):
	print(os.getcwd())
	rclpy.init(args=args)

	coord_transfer = CoordTransfer()

	rclpy.spin(coord_transfer)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	coord_transfer.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	print(os.getcwd())
	main()

