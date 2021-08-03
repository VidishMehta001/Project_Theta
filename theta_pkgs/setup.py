from setuptools import setup

package_name = 'theta_pkgs'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'camera_output = theta_pkgs.camera_output:main'
        ],
    },
)
