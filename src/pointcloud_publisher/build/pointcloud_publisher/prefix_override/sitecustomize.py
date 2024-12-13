import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/yuyifan/ros2_workspace/src/pointcloud_publisher/install/pointcloud_publisher'
