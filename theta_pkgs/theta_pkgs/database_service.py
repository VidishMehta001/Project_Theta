import rclpy
from rclpy.node import Node
import json
import os

from theta_interfaces.srv import Passloc, Takeitem, Checkitem, Taskstatus

class MinimalService(Node):

    def __init__(self):
        super().__init__('item_database_service')
        self.declare_parameter('clear_db', '0')
        self.declare_parameter('threshold', '1')
        self.timer = self.create_timer(10.0, self.save_db)

        self.srv_pass = self.create_service(Passloc, 'pass_loc', self.pass_loc_callback)
        self.srv_check = self.create_service(Checkitem, 'check_item', self.check_item_callback)
        self.srv_take = self.create_service(Takeitem, 'takeitem', self.take_item_callback)
        self.srv_task_status = self.create_service(Taskstatus, 'taskstatus', self.task_status_callback)

        self.db_not_saved = True
        self.accept_updates = True
        self.block_updates_to = 99999
        self.idx_sent = 99999
        self.pose = [0,0,0]

        clear_db = int(self.get_parameter('clear_db').get_parameter_value().string_value)

        ##TODO: Add odom subscriber

        # Load DB
        if clear_db:
            print('deleting DB')
            os.remove('database/database.txt')
        try:
            with open('database/database.txt', 'r') as f:
                db_file = f.read()
                self.db = json.loads(db_file)
            print("Read file...")
            print(self.db)
        except:
            self.db = {}
            print("File failed to read...")
    
    def save_db(self):
        if len(self.db)>1 and self.db_not_saved:
            print('Saving DB...')
            with open('database/database.txt', 'w+') as f:
                f.write(json.dumps(self.db))
            self.db_not_saved = False

    def pass_loc_callback(self, request, response):
        self.threshold = flaot(self.get_parameter('threshold').get_parameter_value().string_value)
        try:
            new_coords = [request.x,request.y,request.z]
            if request.class_id in self.db.keys() and request.class_id != self.block_updates_to:
                for item in self.db[request.class_id]:
                    if dist(item,new_coords)<self.threshold :
                        item = [(item[j]+new_coords[j])/2 for j in range(len(item))]
                        # Quit the entire loop and go to response
                        break
                # Will only reach here is none  of the current coords match
                self.db[request.class_id].append(new_coords)
            elif request.class_id != self.block_updates_to:
                # If DB for this class is empty
                self.db[request.class_id]=[]
                self.db[request.class_id].append(new_coords)
        except:
            response = 0
            return response
        response.accept = 1
        self.db_not_saved = True
        return response

    def check_item_callback(self, request, response):
        if str(request.class_id) in self.db.keys():
            response.size = len(self.db[str(request.class_id)])
        else:
            response.size = 0
        return response

    def take_item_callback(self, request, response):
        if str(request.class_id) in self.db.keys():

            class_items = self.db[str(request.class_id)]
            closest_val = 1000000000000
            for i in range(len(class_items)):
                item = class_items[i]
                if dist(item,self.pose)<closest_val:
                    cloest_idx = i
                    closest_val = dist(item,self.pose)
            response.class_id = request.class_id
            response.x = class_items[cloest_idx][0]
            response.y = class_items[cloest_idx][1]
            response.z = class_items[cloest_idx][2]
            self.idx_sent = cloest_idx
            self.block_updates_to = request.class_id
        else:
            response.class_id = 0
            response.x = 0.0
            response.y = 0.0
            response.z = 0.0
        return response

    def task_status_callback(self, request, response):
        if request.task_completed == 1:
            self.db[self.block_updates_to].pop([self.idx_sent])
        self.block_updates_to = 99999 # May need to change this to something else
        self.idx_sent = 99999 # May need to change this to something else

def dist(a,b):
	a2 = (a[0]**2+a[1]**2+a[2]**2)**0.5
	b2 = (b[0]**2+b[1]**2+b[2]**2)**0.5
	return ((a2-b2)**2)**0.5

def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
