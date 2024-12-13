import pcl
import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import rclpy
from rclpy.node import Node
import sensor_msgs_py.point_cloud2 as pc2  # 用于处理点云数据
from ridar_service_interface.srv import DetectPointCloud
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud2

#定义一个ros节点，用于处理其他ros节点的请求，处理完成之后将数据返回
class RegistrationServer(Node):
    def __init__(self):
        super().__init__('ridar_detection_service')

        # 创建服务
        self.srv = self.create_service(DetectPointCloud, 'ridar_detection_service', self.server_callback)
        
        # 创建订阅者，订阅点云数据
        self.point_cloud_data = []
        self.create_subscription(PointCloud2, 'point_cloud_topic', self.point_cloud_callback, 10)

    def point_cloud_callback(self, msg):

        pc_data_list = []
        # 将 PointCloud2 消息转换为点云数组
        for point in pc2.read_points(msg, skip_nans=True):
            pc_data_list.append([point[0], point[1], point[2]])  # 提取 (x, y, z) 坐标
    
        # 提取 x, y, z 坐标并转换为 NumPy 数组
        points = np.array(pc_data_list)  # 直接将点云数据转换为 NumPy 数组

        # 转换为 np.float64 类型，并调整形状为 (n, 3)
        points = points.astype(np.float64).reshape(-1, 3)

        """处理接收到的服务请求"""
        if points is None:
            self.get_logger().warn("没有接收到点云数据!")
            return None
        
        # 提取地面点和非地面点
        ground_points_pcd, plane_model, non_ground_points_pcd = self.extract_ground_points(points)

        # 拟合地平面并获取平面参数
        a, b, c, d = plane_model#在雷达坐标系下地平面形成的平面的系数
        print(f"平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

        # 获取拟合平面的法向量
        ground_normal = np.array([a, b, c])

        # 计算雷达坐标系到世界坐标系的旋转矩阵和位移
        rotation_matrix, translation = self.calculate_transformation(ground_normal)

        # 显示旋转矩阵和位移
        print("旋转矩阵：\n", rotation_matrix)
        print("平移向量：\n", translation)

        # 筛选出在地平面和距离地平面 40 cm 的平面之间的点云,以及距离雷达距离超出阈值的点云
        selected_points_pcd = self.filter_points_between_planes(non_ground_points_pcd, a, b, c, d, max_height= 0.3, max_distance= 2.8)

        if selected_points_pcd is not None:
            # 对 selected_points_pcd 进行聚类，分离出托盘最前面的横线部分
            front_line_pcd , other_pcd = self.extract_front_line(a, b, c, d, rotation_matrix, translation, selected_points_pcd, eps=0.02, min_samples=8, line_threshold=0.3)
            
            point_start, point_end, midpoint, line_direction = self.fit_line_and_find_extremes(front_line_pcd)

            #已知拟合直线，拟合的地平面的参数，以及中心点，那么我们可以求出托盘的朝向

            # 1.计算垂直于 line_direction 且平行于地面的方向向量
            perpendicular_direction = np.cross(line_direction, np.array([a, b, c]))
            # 2.归一化以确保它是单位向量
            perpendicular_direction = perpendicular_direction / np.linalg.norm(perpendicular_direction)
            print("托盘中心点坐标:",midpoint)
            face_direction = midpoint + perpendicular_direction
            print("托盘的朝向:", face_direction)

            # 应用欧式变换（旋转 + 平移）
            transformed_face_direction = np.dot(rotation_matrix, face_direction) + translation
            print("朝向在 xoy 平面上的分量:", np.array([transformed_face_direction[0], transformed_face_direction[1]]))
            transformed_midpoint = np.dot(rotation_matrix, midpoint) + translation
            # 创建表示直线的线段
            # line_set, point_cloud, cylinder = self.create_line_set(point_start, point_end, midpoint)
            
            # 保存和可视化结果
            if front_line_pcd is not None:
                #将处理之后的数据作为response返回
                self.point_cloud_data.append([[float(transformed_face_direction[0]), float(transformed_face_direction[1])],[float(transformed_midpoint[0]), float(transformed_midpoint[1])]])
            else:
                print("未找到符合条件的托盘横线点云。")
                return None
        else:
            print("未找到符合条件的 selected_points_pcd 点云。")
            return None

    def server_callback(self, request, response):
        print("托盘类型:",request.pallet_types)
        if len(self.point_cloud_data) > 0:
            response_data = self.point_cloud_data.pop(0)
            response.result_1.data = response_data[0]
            response.result_2.data = response_data[1]
            return response
        else:
            response.result_string = "未接受到雷达的点云数据，请等待"
            return response

    def calculate_normal(self, p1, p2, p3):
        # 计算法向量
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # 单位化
        return normal

    def is_parallel_to_ground(self, normal, ground_vector, angle_threshold):
        # 计算法向量和地面向量的夹角
        dot_product = np.dot(normal, ground_vector)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * (180.0 / np.pi)  # 转换为角度
        return angle < angle_threshold

    def filter_by_angle(self, points, angle_threshold=15):
        # 过滤左右范围各30度的点
        filtered_points = []
        for point in points:
            # 计算每个点相对 x 轴的水平夹角（方位角）
            angle = np.degrees(np.arctan2(point[0], point[1]))  # atan2 返回的是弧度，转换为角度
            # print(angle)
            if -angle_threshold+ 5 <= angle <= angle_threshold + 5:
                filtered_points.append(point)
        return np.array(filtered_points)

    def filter_points_between_planes(self, pcd, a, b, c, d, max_height=0.4, max_distance=2.8):
        # 将 PointCloud 对象中的点转换为 numpy 数组
        points = np.asarray(pcd.points)

        # 计算点到地平面的距离
        height = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        
        # 筛选出距离在 [0, max_height] 之间的点，且要求距离为正（即在平面上方）
        mask1 = (height > 0) & (height <= max_height)
        
        points = points[mask1]
        # 计算每个点到原点的欧氏距离
        distance_to_origin = np.linalg.norm(points, axis=1)
        
        # 筛选出距离原点小于 max_distance 的点
        mask2 = distance_to_origin <= max_distance
        
        # 获取同时满足两个条件的点
        selected_points = points[mask2]

        if len(selected_points) != 0:
            # 创建一个 PyVista PolyData 对象
            selected_pcd = pv.PolyData(selected_points)

            # 为所有点设置颜色 (蓝色)
            colors = np.zeros((selected_points.shape[0], 3), dtype=np.uint8)
            colors[:, 0] = 0   # 红色通道
            colors[:, 1] = 0   # 绿色通道
            colors[:, 2] = 255 # 蓝色通道

            # 使用 point_arrays 将颜色应用到点云数据
            selected_pcd.point_arrays['colors'] = colors

            # 返回修改后的点云数据
            return selected_pcd
        else:
            print("没有满足条件的点云数据!")
            return None

    def extract_ground_points(self, points, distance_threshold=0.01):
        # 根据视野角度过滤点
        points_in_view = self.filter_by_angle(points, angle_threshold=20)
        num_points = len(points_in_view)
        print("视野角度内的点个数为:", num_points)
        
        # 确保提取 'x', 'y', 'z' 组件并转换为 np.float64 类型
        points_in_view = np.asarray(points_in_view)
        if points_in_view.dtype.names is not None:
            points_in_view = points_in_view[['x', 'y', 'z']]  # 提取 x, y, z 字段

        points_in_view = points_in_view.view(np.float64).reshape(-1, 3)  # 转换为 3 列数据 (x, y, z)
        filtered_pcd = pv.PolyData(points_in_view)

        # 使用 PyVista 的平面分割方法
        plane_model, inliers = filtered_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )

        # 根据 inliers 分离地面点与非地面点
        ground_points_pcd = filtered_pcd.extract_points(inliers)  # 获取地面点
        non_ground_points_pcd = filtered_pcd.extract_points(inliers, invert=True)  # 获取非地面点

        # 检查是否有地面点
        if ground_points_pcd.n_points != 0:
            print("地面点数量：", ground_points_pcd.n_points)
            return ground_points_pcd, plane_model, non_ground_points_pcd
        else:
            return None, None, None
        
    #计算欧氏变换矩阵
    def calculate_transformation(self, ground_normal):
        # 真实地面法向量 (假设 Z 轴方向)
        world_normal = np.array([0, 1, 0])

        # 计算旋转矩阵：从雷达平面法向量到真实地面法向量
        # 首先，计算旋转轴（法向量的叉积）和旋转角度（法向量的点积）
        axis = np.cross(ground_normal, world_normal)
        sin_angle = np.linalg.norm(axis)
        cos_angle = np.dot(ground_normal, world_normal)

        if sin_angle == 0:
            # 如果法向量完全对齐，不需要旋转
            rotation_matrix = np.eye(3)
        else:
            axis = axis / sin_angle  # 单位化旋转轴
            rotation_matrix = R.from_rotvec(axis * np.arccos(cos_angle)).as_matrix()

        # 计算平移矩阵
        # 如果需要平移矩阵，可以通过已知的点云坐标与世界坐标系的对应关系来计算
        # 假设原点的转换，这里直接使用拟合的平面原点 (centroid) 作为平移量
        centroid = np.mean(ground_normal, axis=0)  # 假设平面法向量的质心为平移点
        translation = centroid

        return rotation_matrix, translation

    #分离出托盘前面部分的线段
    def extract_front_line(self, a, b, c, d, rotation_matrix, translation, selected_points_pcd, eps=0.05, min_samples=5, line_threshold=0.3):
        # 将 PointCloud 对象中的点转换为 numpy 数组
        points = np.asarray(selected_points_pcd.points)

        #高度筛选
        distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        # mask = (distances > line_threshold - 0.03) & (distances < line_threshold + 0.02)
        mask = (distances > (line_threshold - 0.14)) & (distances <= line_threshold)
        points = points[mask]

        # 检查点云数据是否为空
        if points.shape[0] == 0:
            print("No points found after filtering.")
            return None, None  # 如果没有点，直接返回空结果

        print(f"Selected points shape: {points.shape}")  # 输出点云数据的形状

        # 计算新的雷达坐标
        radar_origin = np.array([0, 0, 0])
        new_radar_origin = np.dot(radar_origin, rotation_matrix.T) + translation  # 将雷达原点转换到新的坐标系

        # 使用 DBSCAN 聚类来识别不同的点群
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])  # 仅聚类 x 和 y 维度（假设在水平面上形成线）
        labels = clustering.labels_

        # 获取所有不同的聚类标签，忽略噪声标签 -1，并统计每个聚类的点数
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            print("未识别到任何聚类。")
            return None, None

        label_counts = Counter(labels[labels != -1])  # 统计每个聚类标签的数量
        top_five_labels = [label for label, _ in label_counts.most_common(5)]  # 获取点数最多的五个聚类标签

        print("识别到的聚类数:", len(unique_labels))
        print("点数最多的五个聚类标签:", top_five_labels)

        # 存储每个聚类的原始点云、变换后的点云、解释方差和到雷达原点的距离
        original_clusters = []
        transformed_clusters = []
        distances_to_new_origin = []
        explained_variances = []

        # 遍历数量最多的五个聚类标签
        for label in top_five_labels:
            cluster_points = points[labels == label]
            if len(cluster_points) > 0:
                # 保存原始点云
                original_clusters.append(cluster_points)
                
                # 应用旋转矩阵和平移向量到点云
                transformed_points = np.dot(cluster_points, rotation_matrix.T) + translation
                transformed_clusters.append(transformed_points)
                
                # 计算到新的雷达坐标的平均距离
                distances_avg = np.mean(np.sqrt((transformed_points[:, 0] - new_radar_origin[0])**2 + 
                                                (transformed_points[:, 1] - new_radar_origin[1])**2))
                distances_to_new_origin.append(distances_avg)

                # 计算直线拟合程度（解释方差）
                pca = PCA(n_components=3)
                pca.fit(cluster_points)
                explained_variance_ratio = pca.explained_variance_ratio_
                explained_variances.append(explained_variance_ratio[0])  # 第一主成分的解释方差

        # 找到最直的聚类（解释方差最高的聚类）
        if explained_variances:
            straightest_cluster_index = np.argmax(explained_variances)
            straightest_cluster_points = original_clusters[straightest_cluster_index]  # 使用原始坐标系中的点
            
            # 将最直的聚类转换为 pcl 点云对象
            straightest_cluster_pcd = pv.PolyData(straightest_cluster_points)

            # 为最直的聚类点云设置颜色 (红色)
            colors = np.zeros((straightest_cluster_points.shape[0], 3), dtype=np.uint8)
            colors[:, 0] = 255  # 红色通道
            colors[:, 1] = 0    # 绿色通道
            colors[:, 2] = 0    # 蓝色通道

            # 使用 pcl 点云的 set_colors 方法设置点云的颜色
            straightest_cluster_pcd.point_arrays["colors"] = colors

            # 将其余四个聚类转换为 PointCloud 对象并标记为黄色
            other_clusters_pcds = []
            for i, cluster_points in enumerate(original_clusters):
                if i != straightest_cluster_index:  # 排除最直的聚类
                   # 将最直的聚类转换为 pcl 点云对象
                    other_cluster_pcd = pv.PolyData(cluster_points)

                    # 为最直的聚类点云设置颜色 (红色)
                    colors = np.zeros((cluster_points.shape[0], 3), dtype=np.uint8)
                    colors[:, 0] = 0  # 红色通道
                    colors[:, 1] = 255  # 绿色通道
                    colors[:, 2] = 255   # 蓝色通道

                    # 使用 pcl 点云的 set_colors 方法设置点云的颜色
                    other_cluster_pcd.point_arrays["colors"] = colors
                    other_clusters_pcds.append(other_cluster_pcd)
            
            return straightest_cluster_pcd, other_clusters_pcds
        else:
            print("未找到符合条件的聚类点云。")
            return None, None

    #求线性点云两端的端点以及中心点
    def fit_line_and_find_extremes(self, point_cloud):
        # 将 PointCloud 对象转换为 numpy 数组
        points = np.asarray(point_cloud.points)

        """
        使用密度加权的 PCA 拟合点云主方向。
        
        参数:
        - points: 点云数据，形状为 (N, 3)
        - k: 用于计算密度的近邻数

        返回:
        - principal_direction: 基于密度加权的拟合主方向向量
        - weighted_mean: 密度加权的点云中心
        """
        # Step 1: 计算每个点的密度（基于 k 近邻距离）
        nbrs = NearestNeighbors(n_neighbors=5).fit(points)
        distances, _ = nbrs.kneighbors(points)
        
        # 密度取决于点到其 k 个近邻的平均距离的倒数
        densities = 1 / (np.mean(distances[:, 1:], axis=1) + 1e-6)  # 避免除零

        # Step 2: 计算密度加权的协方差矩阵并进行 PCA
        # 计算加权平均点（加权中心）
        weighted_mean = np.average(points, axis=0, weights=densities)

        # 计算加权协方差矩阵
        centered_points = points - weighted_mean
        weighted_cov_matrix = (densities[:, np.newaxis] * centered_points).T @ centered_points / np.sum(densities)

        # 使用加权协方差矩阵进行 PCA
        eigvals, eigvecs = np.linalg.eigh(weighted_cov_matrix)  # 获取特征值和特征向量
        principal_direction = eigvecs[:, np.argmax(eigvals)]  # 选择最大特征值对应的特征向量作为主方向

        # 将点云投影到主方向上，找到最大和最小的投影点
        projections = points @ principal_direction  # 点云在直线方向上的投影
        min_index = np.argmin(projections)
        max_index = np.argmax(projections)

        # 找到最两端的点
        point_start = points[min_index]
        point_end = points[max_index]

        # 计算两端点的中点
        midpoint = (point_start + point_end) / 2

        return point_start, point_end, midpoint, principal_direction

    #在open3d中显示线段
    def create_line_set(self, point_start, point_end, midpoint):
        # 创建 pcl 点云对象
        line_points = np.array([point_start, point_end, midpoint], dtype=np.float32)
        line_pcd = pv.PolyData(line_points)

        # 为点云设置颜色 (红色)
        colors = np.zeros((line_points.shape[0], 3), dtype=np.uint8)
        colors[:, 0] = 255  # 红色通道
        colors[:, 1] = 0    # 绿色通道
        colors[:, 2] = 0    # 蓝色通道
        line_pcd.point_arrays["colors"] = colors

        # 计算圆柱体的方向向量
        direction = point_end - point_start
        height = np.linalg.norm(direction)  # 圆柱体的高度即两点之间的距离
        direction /= height  # 单位化方向向量

        # 使用 pyvista 创建圆柱体
        cylinder = pv.Cylinder(center=midpoint, direction=direction, radius=0.01, height=height)
        cylinder_mesh = pv.wrap(cylinder)

        # 创建一个可视化窗口并显示点云与圆柱体
        plotter = pv.Plotter()
        plotter.add_mesh(cylinder_mesh, color="black", opacity=1)  # 显示圆柱体
        plotter.add_points(line_points, color="red", point_size=10)  # 显示点云

        plotter.show()

        return line_pcd, cylinder_mesh
    
def main(args=None):
    print("Start Registration Server")
    rclpy.init()
    node = RegistrationServer()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == "__main__":
    main()