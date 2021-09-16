# /right/image_mono
import os
import rclpy
from rclpy.node import Node
import json
from std_msgs.msg import String
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
from operator import add
from statistics import mean
from math import cos, sin
import math
from theta_interfaces.srv import Passloc
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType

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


class CoordsFilterMatcher(Node):

	def __init__(self):
		super().__init__('coord_filter_match')
		self.subscription = self.create_subscription(
			String,
			'/centroid_depth',
			self.model_callback,
			10)
		self.subscription  # prevent unused variable warning

		self.cli = self.create_client(Passloc, 'pass_loc')
		while not self.cli.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('service not available, waiting again...')
		self.req = Passloc.Request()
		self.main_dict = {}
		self.declare_parameter('threshold', '0.8')


	def model_callback(self, msg):
		self.threshold = float(self.get_parameter('threshold').get_parameter_value().string_value)
		data = json.loads(msg.data)
		if len(data)==0:
			return

		new_dict = {}
		for key in data:
			sub_data = data[key]
			new_coords =[]
			if key not in self.main_dict.keys():
				self.main_dict[key]=[]
				for item in sub_data:
					 self.main_dict[key].append(item)
					 self.main_dict[key][-1].append(2)
			else:
				for item in sub_data:
					match_found = False
					for main_item in self.main_dict[key]:
						if match_found:
							continue
						if dist(main_item,item)<self.threshold:
							main_item[0]+=item[0]
							main_item[1]+=item[1]
							main_item[2]+=item[2]
							main_item[3]+=2
							match_found = True
							continue
					if not match_found:
						self.main_dict[key].append(item)
						self.main_dict[key][-1].append(3)

				# Removes if no continous item found
				indexes = []
				for i in range(len(self.main_dict[key])):
					self.main_dict[key][i][3]-=1
					if self.main_dict[key][i][3] == 0:
						indexes.append(i)
					elif self.main_dict[key][i][3] >= 5:
						# This is where it will go to the main database
						self.send_request(self.main_dict[key][i],key)
						indexes.append(i)
						self.get_logger().info("DATABASE NEW UPDATE")
				indexes.reverse()
				for i in indexes:
					self.main_dict[key].pop(i)

	def send_request(self,data,key):
		self.req.class_id = int(key)
		self.req.x = float(data[0])
		self.req.y = float(data[1])
		self.req.z = float(data[2])
		accept = self.cli.call_async(self.req)
		self.get_logger().info(str(accept))
def dist(a,b):
	a2 = (a[0]/a[3]**2+a[1]/a[3]**2+a[2]/a[3]**2)**0.5
	b2 = (b[0]**2+b[1]**2+b[2]**2)**0.5
	return ((a2-b2)**2)**0.5


def main(args=None):
	print(os.getcwd())
	rclpy.init(args=args)

	coord_filter_match = CoordsFilterMatcher()

	rclpy.spin(coord_filter_match)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	coord_filter_match.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	print(os.getcwd())
	main()

