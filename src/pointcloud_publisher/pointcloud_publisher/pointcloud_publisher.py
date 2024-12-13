import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import open3d as o3d
from sensor_msgs.msg import PointField
import struct
import os

class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('PointCloudPublisher')

        # 创建发布者，将点云数据发布到 point_cloud_topic
        self.publisher = self.create_publisher(PointCloud2, 'point_cloud_topic', 10)
        
        # 从目录读取 PCD 文件并转换为 PointCloud2 格式
        self.pcd_dir = '/home/cqxx/ros2_workspace/ridar'  # 指定目录
        self.pcd_files = [f for f in os.listdir(self.pcd_dir) if f.endswith('.pcd')]
        self.pcd_index = 0  # 当前读取的 PCD 文件索引
        
        # 定时器，周期性发布下一个点云
        self.timer = self.create_timer(1.0, self.publish_next_point_cloud)

    def publish_next_point_cloud(self):
        # 循环读取目录中的 PCD 文件并发布
        if self.pcd_index >= len(self.pcd_files):
            self.pcd_index = 0  # 如果读取到最后一个文件，则重新从头开始
        
        pcd_file = self.pcd_files[self.pcd_index]
        pcd_file_path = os.path.join(self.pcd_dir, pcd_file)
        self.get_logger().info(f'Reading PCD file: {pcd_file_path}')
        
        pcd = o3d.io.read_point_cloud(pcd_file_path)
        points = pcd.points
        
        # Convert open3d point cloud to ROS PointCloud2 message
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'  # 你可以根据实际情况修改 frame_id
        ros_cloud = self.create_point_cloud2_message(header, points)
        
        # 发布点云数据到 topic
        self.get_logger().info(f'Publishing point cloud data from {pcd_file}')
        self.publisher.publish(ros_cloud)
        
        # 增加文件索引
        self.pcd_index += 1

    def create_point_cloud2_message(self, header, points):
        # 将 open3d 点云数据转换为 ROS PointCloud2 格式的消息
        field_names = ['x', 'y', 'z']
        fields = [
            PointField(name=field, offset=i * 4, datatype=PointField.FLOAT32, count=1) 
            for i, field in enumerate(field_names)
        ]
        
        # 使用 struct 来转换数据格式
        pc_data = []
        for point in points:
            pc_data.append(struct.pack('fff', *point))

        # 将点云数据转换为 ROS 消息
        pc_data = b''.join(pc_data)
        return PointCloud2(
            header=header,
            height=1,
            width=len(points),
            is_dense=True,
            is_bigendian=False,
            point_step=12,  # 每个点3个float（x,y,z）
            row_step=12 * len(points),
            fields=fields,
            data=pc_data
        )

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
