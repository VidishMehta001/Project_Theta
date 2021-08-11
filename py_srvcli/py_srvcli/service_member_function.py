from example_interfaces.srv import AddTwoInts

import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)
        self.df = pd.read_csv('general/data.csv')

    def add_two_ints_callback(self, request, response):
        response.sum = int(self.df.loc[0,'item']) #request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))

        return response


def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
