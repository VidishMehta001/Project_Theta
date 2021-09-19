from setuptools import setup
import os
package_name = 'theta_pkgs'
from glob import glob
from setuptools import find_packages
packages=find_packages(exclude=['test'])

setup(
    name=package_name,
    version='0.0.0',
    packages=['theta_pkgs','theta_pkgs/utils','theta_pkgs/object_detection','theta_pkgs/models'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mecarill',
    maintainer_email='mihkailkennerley@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'model_inference = theta_pkgs.model_inference:main',
            'depth_coord = theta_pkgs.depth_coord_extract:main',
            'coords_filter_match = theta_pkgs.coords_filter_match:main',
            'database_service = theta_pkgs.database_service:main',
            'camera_output = theta_pkgs.camera_output:main',
            'last_mile = theta_pkgs.last_mile:main'
        ],
    },
)
