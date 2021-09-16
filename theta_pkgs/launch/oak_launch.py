from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='depthai_examples',
            executable='rgb_stereo_node',
            name='stereo_node'
        ),
        Node(
            package='image_proc',
            executable='image_proc',
            name='image_proc',
            remappings=[
                ('image_raw', 'right/image'),
                ('camera_info', 'right/camera_info'),
                ('image_mono', 'right/image_mono'),
                ('image_rect', 'right/image_rect'),
            ]
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
