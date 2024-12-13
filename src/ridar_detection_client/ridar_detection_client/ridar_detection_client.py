import rclpy
from rclpy.node import Node
from ridar_service_interface.srv import DetectPointCloud  # 请根据实际情况修改包名和服务名称

class PointCloudClient(Node):
    def __init__(self):
        super().__init__('pointcloud_client')
        self.cli = self.create_client(DetectPointCloud, 'ridar_detection_service')

        # 等待服务可用
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.get_logger().info('Service found, sending request...')
        
        # 创建服务请求对象
        request = DetectPointCloud.Request()
        
        # 设置托盘的种类为 1
        request.pallet_types = 1  # 根据需要设置种类标识

        # 发送请求并获取 future
        self.future = self.cli.call_async(request)
        
        # 等待并处理响应
        self.future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        try:
            # 检查是否有响应
            response = future.result()
            if response.success.data == True:
                self.get_logger().info(f'Received response: {response}')
            else:
                pass 
                # self.get_logger().error('No response received.')

        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

        # 继续发送下一个请求
        self.send_next_request()

    def send_next_request(self):
        # 创建新的请求
        request = DetectPointCloud.Request()
        request.pallet_types = 1  # 修改为需要的托盘种类
        
        # 发送下一个请求
        self.future = self.cli.call_async(request)
        self.future.add_done_callback(self.response_callback)

def main(args=None):
    rclpy.init(args=args)
    client = PointCloudClient()

    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
