import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge

class CameraPub(Node):
	def __init__(self):
		self.get_logger().info("Starting camera_pub Node...")
		self.get_logger().info("Publishing to /right/image_mono...")
		super().__init__('camera_pub')
		self.publisher_ = self.create_publisher(Image, '/right/image_mono', 10)
		self.bridge = CvBridge()
		self.ImgRead()

	def ImgRead(self):
		cap = cv.VideoCapture(0)
		cap.set(cv.CAP_PROP_BUFFERSIZE, 2)
		while cap.isOpened and i<1000:
			ret_val, img = cap.read()
			img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
			if ret_val:
				image_message = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
				self.publisher_.publish(image_message) 
				self.get_logger().info("Published Webcam Image...")

def main(args=None):
	rclpy.init(args=args)

	camera_pub = CameraPub()

	rclpy.spin(camera_pub)

	camera_pub.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()

