from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='theta_pkgs',
            executable='camera_output',
            name='camera_output'
        ),
        Node(
            package='theta_pkgs',
            executable='model_inference',
            name='model_inference'
        ),
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_image_view'
        )
    ])
