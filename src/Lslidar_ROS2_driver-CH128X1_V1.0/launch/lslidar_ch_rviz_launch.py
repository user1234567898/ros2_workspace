#!/usr/bin/python3
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument

import lifecycle_msgs.msg
import os
import subprocess
import sys


def generate_launch_description():
    driver_dir = os.path.join(get_package_share_directory('lslidar_ch_driver'), 'params', 'lslidar_ch.yaml')
    rviz_dir = os.path.join(get_package_share_directory('lslidar_ch_driver'), 'rviz_cfg', 'lslidar_ch_driver.rviz')

    p = subprocess.Popen("echo $ROS_DISTRO", stdout=subprocess.PIPE, shell=True)
    driver_node = ""
    rviz_node = ""
    ros_version = p.communicate()[0]
    print(ros_version)
    if ros_version == b'dashing\n' or ros_version == b'eloquent\n':
        print("ROS VERSION: dashing/eloquent")
        driver_node = LifecycleNode(package='lslidar_ch_driver',
                                    node_namespace='CH',
                                    node_executable='lslidar_ch_driver_node',
                                    node_name='lslidar_ch_driver_node',
                                    output='screen',
                                    parameters=[driver_dir],
                                    )
        
        rviz_node = Node(
            package='rviz2',
            node_namespace='ch128x1',
            node_executable='rviz2',
            node_name='rviz2',
            arguments=['-d', rviz_dir],
            output='screen')
        
    elif ros_version == b'foxy\n' or ros_version == b'galactic\n' or ros_version == b'humble\n':
        print("ROS VERSION: foxy/galactic/humble")
        driver_node = LifecycleNode(package='lslidar_ch_driver',
                                    namespace='CH',
                                    executable='lslidar_ch_driver_node',
                                    name='lslidar_ch_driver_node',
                                    output='screen',
                                    emulate_tty=True,
                                    parameters=[driver_dir],
                                    )
        
        rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_dir],
            output='screen')
        
    else:
        print("Please configure the ros environment")
        sys.exit()

    return LaunchDescription([
        driver_node,rviz_node
    ])
