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
		self.subscription_odom = self.create_subscription(
			Odometry,
			'/odometry',
			self.model_callback_odom,
			10)
		self.subscription2  # prevent unused variable warning
		self.image_data  = []
		self.bridge = CvBridge()

		self.right_proj_matrix = np.matrix([
			[917.723107, 0.000000, 647.542511, -67.342795],
			[0.000000, 917.723107, 362.991959, 0.000000],
			[0.000000, 0.000000, 1.000000, 0.000000]])
		self.inv_right_proj_matrix = self.right_proj_matrix.getI()

		self.color_proj_matrix = np.matrix([
			[1479.458984, 0.000000, 950.694458, 0.0],
			[0.0, 1477.587158, 530.697632, 0.0], 
        	[0.0, 0.0, 1.0, 0.0]])

		self.inv_color_proj_matrix = self.color_proj_matrix.getI()

		self.pose_pos =[0,0,0]
		self.pose_ori = [0,0,0]

	def model_callback(self, msg):
		data = json.loads(msg.data)
		if len(self.image_data)==0 or len(data)==0:
			return

		image_shape = (self.image_data.shape[0]-1,self.image_data.shape[1]-1)
		new_coords =[]
		try:
			data = data['1']
		except:
			print(data)
			return
		for item in data:
			# When using colour image Colour_intrinsic > Extrinsic > Right_Mono_intrinsic
			item.append(1)
			print(item)
			loc_matrix = np.matrix(item).getT()
			#print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<',loc_matrix)
			real_coords = np.matmul(self.inv_color_proj_matrix,loc_matrix)
			matrix_val = np.matmul(self.right_proj_matrix,real_coords)
			#print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',matrix_val)
			x, y = int(matrix_val[0]), int(matrix_val[1])
			if x > 720 or y >1280 or x<0 or y <0:
				continue
			z_val_box = self.image_data[x-3:x+3,y-3:y+3,0]
			try:
				z_val = mean([item for sublist in z_val_box for item in sublist if item!=0 and item<15000])
			except:
				print(z_val_box)
				print('NaN Error...')
				continue
			location = list([float(real_coords[0])*z_val, float(real_coords[1])*z_val,float(real_coords[2])*z_val])


			# Rotate X, Y and Z by self.pose_ori values
			#TODO

			# Translate X,Y,Z by self.pose_pos values
			location = list(map(add,location,self.pose_pos))
			new_coords.append(location)
			print("----------------------------------")

		new_msg = String()
		new_msg.data = json.dumps(new_coords, cls=NpEncoder)

		self.get_logger().info('Got coords...')
		self.publisher_.publish(new_msg)
		
	def model_callback2(self, msg):
		image_np=self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
		if len(image_np.shape)==2:
			img2 = cv.merge((image_np,image_np,image_np))
			image_np = img2
		self.image_data = image_np

	def model_callback_odom(self, msg):
		pose_pos1 = msg.data.pose.pose.position
		self.pose_pos =[pose_pos1.x,pose_pos1.y,pose_pos1.z]

		pose_ori_quat = msg.data.pose.pose.orientation
		self.pose_ori = euler_from_quaternion(pose_ori_quat.x,pose_ori_quat.y,pose_ori_quat.z,pose_ori_quat.w)

def euler_from_quaternion(x, y, z, w):
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + y * y)
	roll_x = math.atan2(t0, t1)

	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	pitch_y = math.asin(t2)

	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (y * y + z * z)
	yaw_z = math.atan2(t3, t4)

	return [roll_x, pitch_y, yaw_z]# in radians

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

