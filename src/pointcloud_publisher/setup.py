from setuptools import find_packages, setup

package_name = 'pointcloud_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'open3d',      
        'numpy',       
    ],
    zip_safe=True,
    maintainer='yuyifan',
    maintainer_email='yuyifan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointcloud_publisher = pointcloud_publisher.pointcloud_publisher:main'
        ],
    },
)
