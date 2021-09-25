
from flask import Flask, jsonify
import requests
import time
import roslibpy
import json
import math

# Setting up Flask
app = Flask(__name__)
mycroft_host = 'http://soltistech.ddns.net:5000'
host = 'http://192.168.10.121:5001'

current_pos=[0,0,0]
#TODO hold dict of asked items (only 1 per item)

# Setting up ROS Bridge
client = roslibpy.Ros(host='localhost', port=9090)
client.run()

last_send_nav=time.time()

# Mycroft server requests for class_id position and service returns x,y,z
# class_id of 999 means item is not found (Service Request)
@app.route("/findtest/<objectName>")
def findtest(objectName):
	service_takeitem = roslibpy.Service(client, '/takeitem', 'theta_interfaces/takeitem')
	request = roslibpy.ServiceRequest()
	request['class_id'] = int(objectName)
	print('Calling service...')
	result = service_takeitem.call(request)
	print("Got TakeItem service results...")
	class_id = result['class_id']
	x = result['x']
	y = result['y']
	z = result['z']
	if z == 0 and y==0 and x==0:
		class_id = 999

	# Wen Cheng to take x,y,z and plan a path to return distance & add 'dist':float
	# If not able to use bird fly dist

	requests.get(mycroft_host+'/find/'+str(class_id)+'/'+str(json.dumps({"class_id":class_id,"x":x,"y":y,"z":z}))+'/0')
	print(str(json.dumps({"class_id":class_id,"x":x,"y":y,"z":z})))
	# Web Page Visualisation
	return jsonify({"class_id":class_id,"x":x,"y":y,"z":z}) 

# Sending goal pose information (Topic Publisher)
@app.route("/startnav/<objectName>")
def startnav(objectName, seq):
	talker_pose = roslibpy.Topic(client, '/goal_pose', 'geometry_msgs/PoseStamped')
	seq = json.loads(seq)
	msg = roslibpy.Message()
	msg["header"]["stamp"]["sec"]=0
	msg["header"]["frame_id"]="map"
	msg["pose"]["position"]["x"]=seq["x"]
	msg["pose"]["position"]["y"]=seq["y"]
	msg["pose"]["position"]["z"]=seq["z"]
	msg["pose"]["orientation"]["1"]=1.0
	if client.is_connected:
		talker_pose.publish(msg)

	return jsonify(success=True)

# # Recieving movement commands (Topic Listener)
# def cmd_vel_callback(message):
# 	front = message["linear"]["x"]
# 	turn = message["angular"]["z"]
# 	if front > 0:
# 		direction = 'forward'
# 	else:
# 		direction = 'backward'
# 	#requests.get(mycroft_host+'/turn/'+direction)
# 	print(direction)
# listener_cmd_vel = roslibpy.Topic(client, '/cmd_vel', 'geometry_msgs/Twist')
# listener_cmd_vel.subscribe(cmd_vel_callback)
# cmd_vel_list = []

#Receive path for turning
def path_callback(message):
	# global current_pos
	# x_p=message["poses"][-1]["pose"]["position"]["x"]
	# y_p=message["poses"][-1]["pose"]["position"]["y"]
	# z_p=message["poses"][-1]["pose"]["position"]["z"]
	# print ('x_path:'+str(x_p))
	# print ('y_path:'+str(y_p))

	global last_send_nav

	x_p=message["point"]["x"]
	y_p=message["point"]["y"]


	now=time.time()

	if int(now-last_send_nav)>2:

		if abs(y_p)>0.18:
			if y_p>0:
				print ("left")
				requests.get(mycroft_host+'/turn/'+'left')
			else:
				print ("right")
				requests.get(mycroft_host+'/turn/'+'right')
		else:
			print ("Walk Straight")
			requests.get(mycroft_host+'/turn/'+'straight')
		last_send_nav=time.time()



listener_path = roslibpy.Topic(client, '/lookahead_point', 'geometry_msgs/PointStamped')
listener_path.subscribe(path_callback)

#check for current pose
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
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
     
        return roll_x, pitch_y, yaw_z # in radians

def odom_callback(message):
	if message["header"]["frame_id"]=="odom":
		x_o=message["pose"]["pose"]["position"]["x"]
		y_o=message["pose"]["pose"]["position"]["y"]

		x_c=message["pose"]["pose"]["orientation"]["x"]
		y_c=message["pose"]["pose"]["orientation"]["y"]
		z_c=message["pose"]["pose"]["orientation"]["z"]
		w_c=message["pose"]["pose"]["orientation"]["w"]
		(x_a, y_a, z_a) = euler_from_quaternion (x_c, y_c, z_c, w_c)
		# global current_pos
		# current_pos[0]=x_o
		# current_pos[1]=y_o
		# current_pos[2]=z_a

	#goal pose supposed to come from earlier node
	if abs(current_pos[0]-x_goal)<0.25 and abs(current_pos[1]-y_goal)<0.25:
		print ("goal reached")
		requests.get(mycroft_host+'/goal_reached/')


listener_odom = roslibpy.Topic(client, '/odom', 'nav_msgs/Odometry')
listener_odom.subscribe(odom_callback)


@app.route("/cancelnav")
def cancel_nav():
	cancel_pub = roslibpy.Topic(client, '/cancel_nav', 'std_msgs/String')
	cancel_pub.publish(roslibpy.Message({'data': 'cancel'}))