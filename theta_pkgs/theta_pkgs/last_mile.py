import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge
import mediapipe as mp
import json
from statistics import mean
from time import time
from std_srvs.srv import SetBool

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class LastMile(Node):
	def __init__(self):
		super().__init__('last_mile')
		self.get_logger().info("Starting last_mile Node...")
		self.subscription = self.create_subscription(
			Image,
			'/color/image',
			self.image_callback,
			10)

		self.subscription_coord = self.create_subscription(
			String,
			'/centroid',
			self.centroid_callback,
			10)
		self.subscription_depth = self.create_subscription(
			Image,
			'/stereo/depth',
			self.depth_callback,
			10)
		self.subscription_depth = self.create_subscription(
			Image,
			'/right/image',
			self.right_callback,
			10)
		self.centroid = []
		self.depth_image = []
		self.palm = []
		self.palm_coords = []
		self.centroid_coords = []
		self.centroid_time = 0
		self.palm_time = 0
		self.publisher_ = self.create_publisher(Image, '/color/image_hands', 10)
		self.publisher_lastmile_ = self.create_publisher(String, '/lastmile', 10)
		self.srv_pass = self.create_service(SetBool, 'last_mile_switch', self.last_mile_switch)
		self.publisher_depth_ = self.create_publisher(Image, '/image_depth_dots', 10)
		self.bridge = CvBridge()
		self.item = 0 # class_id of item to track
		self.subscription  # prevent unused variable warning
		self.subscription_coord  # prevent unused variable warning
		self.node_active = False
		self.hands = mp_hands.Hands(
				max_num_hands=1,
				min_detection_confidence=0.5,
				min_tracking_confidence=0.5)

		self.right_proj_matrix = np.matrix([
			[442.960220, 0.000000, 360.386665, -32.884516],
			[0.000000, 442.960220, 196.041361, 0.000000],
			[0.000000, 0.000000, 1.000000, 0.000000]])
		self.inv_right_proj_matrix = self.right_proj_matrix.getI()

		self.color_proj_matrix = np.matrix([
			[1479.458984, 0.000000, 950.694458, 0.0],
			[0.0, 1477.587158, 530.697632, 0.0], 
        	[0.0, 0.0, 1.0, 0.0]])

		self.inv_color_proj_matrix = self.color_proj_matrix.getI()

		self.dist_thresholds = [0.1,0.1,0.1]

	def image_callback(self,image):
		image=self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
		if len(image.shape)==2:
			img2 = cv.merge((image,image,image))
			image = img2
		#image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
		image.flags.writeable = False
		results = self.hands.process(image)
		image.flags.writeable = True
		image_height, image_width, _ = image.shape
		image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				mp_drawing.draw_landmarks(
					image,
					hand_landmarks,
					mp_hands.HAND_CONNECTIONS,
					mp_drawing_styles.get_default_hand_landmarks_style(),
					mp_drawing_styles.get_default_hand_connections_style())
				top_palm = [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width,
					hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height]
				bottom_palm = [hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width,
					hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height]
				self.palm = [(top_palm[0]+bottom_palm[0])/2,(top_palm[1]+bottom_palm[1])/2]
				self.palm_time = time()
		else:
			self.palm = []
		image_message = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
		self.publisher_.publish(image_message)
		self.find_distance()

	def centroid_callback(self,msg):
		data = json.loads(msg.data)
		if data is None:
			return
		if str(self.item) in data.keys():
			self.centroid = data[str(self.item)][0]
			self.centroid_time = time()
		else:
			self.centroid = []

	def depth_callback(self,image):
		image_np=self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
		if len(image_np.shape)==2:
			img2 = cv.merge((image_np,image_np,image_np))
			image_np = img2
		self.depth_image = image_np

	def right_callback(self,image):
		image_np=self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
		if len(image_np.shape)==2:
			img2 = cv.merge((image_np,image_np,image_np))
			image_np = img2
		self.right_image = image_np

	def find_distance(self):
		if not self.node_active:
			return
		msg = String()
		if time()-self.centroid_time>1:
			self.centroid_coords = []
		else:
			msg.data = "no item"

		if time()-self.palm_time>1:
			self.palm_coords = []
		else:
			msg.data = "no palm"

		print(len(self.centroid))
		if len(self.palm)==2:
			coords_check = self.getCoords(self.palm,1)
			if len(coords_check)==3:
				self.palm_coords = coords_check+np.array([0.1,0,0])
				self.palm_time = time()
		if len(self.centroid)==2:
			coords_check = self.getCoords(self.centroid,0)
			if len(coords_check)==3:
				self.centroid_coords = coords_check
				self.centroid_time = time()
		print(self.palm_coords)
		print(self.centroid_coords)

		try:
			distance = np.array(self.centroid_coords)-np.array(self.palm_coords)
			print(distance)
			msg.data = self.return_movement(distance)
		except:
			print(".")
		self.publisher_lastmile_.publish(msg)
		image_message = self.bridge.cv2_to_imgmsg(np.array(self.right_image),  encoding="rgb8")
		self.publisher_depth_.publish(image_message)
		print('-----------------------------')

	def getCoords(self,item,palm_set):
		item = np.hstack((item,1))
		loc_matrix = np.matrix(item).getT()
		real_coords = np.matmul(self.inv_color_proj_matrix,loc_matrix)
		matrix_val = np.matmul(self.right_proj_matrix,real_coords)
		x, y = int(matrix_val[0]), int(matrix_val[1])
		if x > 640 or y >400 or x<0 or y <0:
			print("error")
			print(x,y)
			return []
		x-=50
		if palm_set:
			x-=50
		z_val_box = self.depth_image[y-3:y+3,x-3:x+3,0]
		self.right_image = cv.circle(self.right_image, (x,y), 20, (255,0,0), 4)
		try:
			z_val = mean([item for sublist in z_val_box for item in sublist if item!=0 and item<15000])
		except:
			print('NaN Error...')
			return []
		location_mat = np.array([float(real_coords[2]),float(real_coords[0]),float(real_coords[1])])*z_val/1000
		return location_mat

	def return_movement(self,distance):
		if distance[0]>self.dist_thresholds[0]:
			return "forwards"
		elif distance[0]<-self.dist_thresholds[0]:
			return "backwards"
		elif distance[2]>self.dist_thresholds[2]:
			return "down"
		elif distance[2]<-self.dist_thresholds[2]:
			return "up"
		elif distance[1]>self.dist_thresholds[1]:
			return "right"
		elif distance[1]<-self.dist_thresholds[1]:
			return "left"
		else:
			return "complete"

	def last_mile_switch(self, request, response):
		self.node_active = request.data
		response.success = True
		response.message = "Last Mile node has been updated to: "+str(self.node_active)
		return response

def main(args=None):
	rclpy.init(args=args)

	last_mile = LastMile()

	rclpy.spin(last_mile)

	last_mile.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()

