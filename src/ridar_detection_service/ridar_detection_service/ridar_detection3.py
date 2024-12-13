import open3d as o3d
import numpy as np
import math
import json
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import rclpy
from rclpy.node import Node
import sensor_msgs_py.point_cloud2 as pc2  # 用于处理点云数据
from   ridar_service_interface.srv import DetectPointCloud
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud2
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import threading 


class VerticalPlaneModel(BaseEstimator, RegressorMixin):
    """
    自定义平行于 z 轴的平面模型：ax + by + d = 0
    """
    def fit(self, X, y):
        # 构造设计矩阵 [x, y, 1]
        A = np.hstack((X, np.ones((X.shape[0], 1))))  # 设计矩阵
        plane_params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)  # 最小二乘拟合
        self.a, self.b, self.d = plane_params  # 提取平面参数
        return self

    def predict(self, X):
        # 预测点是否符合平面方程（忽略 z 坐标）
        return -(self.a * X[:, 0] + self.b * X[:, 1] + self.d)

def fit_vertical_plane_ransac(pointcloud):
    """
    使用 RANSAC 拟合平行于 z 轴的平面
    :param pointcloud: 点云数据，形状为 Nx3 的 numpy 数组
    :return: 平面参数 (a, b, d)
    """
    # 提取 x 和 y 坐标，忽略 z
    X = pointcloud[:, :2]  # 取 x 和 y 坐标
    y = np.zeros(X.shape[0])  # z = 0 的假设（平行于 z 轴）

    # 使用 RANSAC 拟合平面
    model = RANSACRegressor(
        estimator=VerticalPlaneModel(),
        min_samples=3,               # 至少需要 3 个点
        residual_threshold=0.01,     # 平面内点的距离阈值
        random_state=42
    )
    model.fit(X, y)

    # 获取平面参数
    a, b, d = model.estimator_.a, model.estimator_.b, model.estimator_.d
    print(a)
    print(b)
    print(d)
    return a, b, d

#定义一个ros节点，用于处理其他ros节点的请求，处理完成之后将数据返回
class RegistrationServer(Node):
    def __init__(self):
        super().__init__('ridar_detection_service')
        #读取配置文件
        with open("/home/cqxx/ros2_workspace/src/ridar_detection_service/config/config1.json", 'r') as file:
            config = json.load(file) 

        # 创建服务
        self.srv = self.create_service(DetectPointCloud, 'ridar_detection_service', self.server_callback)
        
        # 创建订阅者，订阅点云数据
        self.point_cloud_data = []
        self.latest_msg = None  # 保存最新的消息
        self.point_cloud_num = 0
        self.processing = False  # 标志当前是否在处理消息
        # 创建线程锁
        self.lock = threading.Lock()

        #加载托盘相关参数
        self.detection_max_height = config["detection_max_height"]
        self.detection_max_distance = config["detection_max_distance"]
        self.detection_min_distance = config["detection_min_distance"]
        self.extract_front_line_min_samples = config["extract_front_line_min_samples"]
        self.extract_front_line_line_max_threshold = config["extract_front_line_line_max_threshold"]
        self.extract_front_line_line_min_threshold = config["extract_front_line_line_min_threshold"]
        self.pallet_width = config["pallet_width"]

        # self.executor_pool = concurrent.futures.ThreadPoolExecutor()  # 创建线程池
        self.create_subscription(PointCloud2, '/ch128x1/lslidar_point_cloud', self.point_cloud_callback, 10)

    def point_cloud_callback(self, msg):
        """将接收到的点云消息保存为最新数据"""
        with self.lock:  # 确保线程安全
            self.latest_msg = msg
        
        # 如果没有正在处理的任务，直接启动处理
        if not self.processing:
            threading.Thread(target=self.process_point_cloud).start()

    def process_point_cloud(self):

        """处理点云数据"""
        while True:
            with self.lock:
                # 获取最新消息并清空
                msg = self.latest_msg
                self.latest_msg = None

            if msg is None:
                # 如果没有新的消息需要处理，退出循环
                self.processing = False
                break
        
            self.processing = True  # 标记正在处理数据
            pc_data_list = []
            # 将 PointCloud2 消息转换为点云数组
            for point in pc2.read_points(msg, skip_nans=True):
                pc_data_list.append([point[0], point[1], point[2]])  # 提取 (x, y, z) 坐标
        
            # 提取 x, y, z 坐标并转换为 NumPy 数组
            points = np.array(pc_data_list)  # 直接将点云数据转换为 NumPy 数组

            # 转换为 np.float64 类型，并调整形状为 (n, 3)
            points = points.astype(np.float64).reshape(-1, 3)

            # self.save_point_cloud_as_ply(points, f"/home/cqxx/ros2_workspace/point_cloud_data/point_cloud{self.point_cloud_num}.ply")
            # self.point_cloud_num = self.point_cloud_num + 1

            """处理接收到的服务请求"""
            if points is None:
                self.get_logger().warn("没有接收到点云数据!")
                return 
            
            # 提取地面点和非地面点
            ground_points_pcd, plane_model, non_ground_points_pcd = self.extract_ground_points(points)

            # o3d.visualization.draw_geometries([ground_points_pcd])
            # 拟合地平面并获取平面参数
            a, b, c, d = plane_model#在雷达坐标系下地平面形成的平面的系数
            print(f"平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

            # 获取拟合平面的法向量
            ground_normal = np.array([a, b, c])

            # 计算雷达坐标系到世界坐标系的欧式变换矩阵
            transformation_matrix_inv = self.compute_inverse_transformation_matrix(ground_normal,radar_translation = [0,0,0])#表示雷达坐标系到世界坐标系没有偏移值

            # 筛选出在地平面和距离地平面 30 cm 的平面之间的点云,以及距离雷达距离超出阈值的点云
            selected_points_pcd = self.filter_points_between_planes(non_ground_points_pcd, a, b, c, d)

            # front_line_points_clouds, middle_point, front_line_direction = self.detect_pallet_feet_with_orientation(selected_points_pcd,a,b,c,d)

            # perpendicular_direction1 = np.cross(front_line_direction, np.array([a, b, c]))
            # perpendicular_direction1 = perpendicular_direction1 / np.linalg.norm(perpendicular_direction1)

            # # 计算夹角的余弦值
            # cos_theta = perpendicular_direction1[1] / math.sqrt(perpendicular_direction1[0]**2 + perpendicular_direction1[1]**2)
            # # 求夹角（弧度）
            # angle_rad = math.acos(cos_theta)
            # # 可选：将弧度转换为角度（度数）
            # angle_deg = math.degrees(angle_rad)

            # print("根据托盘立柱求出的托盘中心点坐标(雷达坐标系):",middle_point[0],middle_point[1])
            # print("根据托盘立柱求出的托盘朝向向量(雷达坐标系):",perpendicular_direction1)
            # print("根据托盘立柱求出的夹角:",angle_deg)
            # o3d.visualization.draw_geometries([front_line_points_clouds])

            if selected_points_pcd is not None:
                # 对 selected_points_pcd 进行聚类，分离出托盘最前面的横线部分
                front_line_pcd , other_pcd = self.extract_front_line(a, b, c, d, transformation_matrix_inv, selected_points_pcd)
                if front_line_pcd is None:
                    continue
                
                left_point, right_point, midpoint, line_direction, extended_pcd = self.fit_line_and_find_extremes(front_line_pcd, non_ground_points_pcd)

                if line_direction is None:
                    continue
                predict_midpoint = None
                #已知拟合直线，拟合的地平面的参数，以及中心点，那么我们可以求出托盘的朝向

                # 1.计算垂直于 line_direction 且平行于地面的方向向量
                perpendicular_direction2 = np.cross(line_direction, np.array([a, b, c]))
                # 2.归一化以确保它是单位向量
                perpendicular_direction2 = perpendicular_direction2 / np.linalg.norm(perpendicular_direction2)
                # 计算夹角的余弦值
                sin_theta = perpendicular_direction2[0] / math.sqrt(perpendicular_direction2[0]**2 + perpendicular_direction2[1]**2)
                # 求夹角（弧度）
                angle_rad = math.asin(sin_theta)
                # 可选：将弧度转换为角度（度数）
                angle_deg = math.degrees(angle_rad)

                #增加一个逻辑，当托盘的x轴偏差很大的时候,只能检测到部分横梁，这个时候需要自主推测中心点的位置
                print("拟合线段的长度:", np.linalg.norm(left_point - right_point))
                if np.linalg.norm(left_point - right_point) < self.pallet_width - 0.1: #1为托盘的宽度
                    #当前端横梁被截断的时候中心点一定偏向被截断的那一边
                    if  angle_deg > 0 and line_direction[0] < 0:
                        print(f"aaaaaaaaaaaaaaa,left_point{left_point},right_point{right_point}")
                        predict_midpoint = left_point - (self.pallet_width/2) * (line_direction / np.linalg.norm(line_direction))

                    if angle_deg < 0 and line_direction[0] < 0:
                        print(f"bbbbbbbbbbbbbbbbb,left_point{left_point},right_point{right_point}")
                        predict_midpoint = right_point + (self.pallet_width/2) * (line_direction / np.linalg.norm(line_direction))
                else:
                    print(f"ccccccccccccccccc,left_point{left_point},right_point{right_point}")

                # 保存和可视化结果
                if front_line_pcd is not None:
                    # o3d.visualization.draw_geometries([non_ground_points_pcd, front_line_pcd] + other_pcd)
                    #将处理之后的数据作为response返回
                    """更新缓存队列，确保队列长度为 1"""
                    with self.lock:  # 使用线程锁保护
                        if len(self.point_cloud_data) > 0:
                            self.point_cloud_data.pop(0)  # 丢弃旧数据
                        if predict_midpoint is None:
                            print("托盘中心点坐标(雷达坐标系):",midpoint[0],midpoint[1])
                            print("托盘的朝向(雷达坐标系):", perpendicular_direction2)
                            print("夹角：",angle_deg)
                            self.point_cloud_data.append([midpoint[0], midpoint[1],angle_deg,True])
                        else:
                            print("托盘中心点坐标预测值(雷达坐标系):",predict_midpoint[0], predict_midpoint[1])
                            print("托盘的朝向(雷达坐标系):", perpendicular_direction2)
                            print("夹角：",angle_deg)
                            self.point_cloud_data.append([predict_midpoint[0], predict_midpoint[1],angle_deg,True])
                else:
                    print("未找到符合条件的托盘横线点云。")
                    continue 
            else:
                print("未找到符合条件的 selected_points_pcd 点云。")
                continue 
        

    # 定义函数
    def increasing_function(self, imput, k=1):
        return 1 - np.exp(-k * imput)

    def server_callback(self, request, response):
        print("托盘类型:",request.pallet_types)
        with self.lock:
            if len(self.point_cloud_data) > 0:
                response_data = self.point_cloud_data.pop(0)
                response.x.data = response_data[0]
                response.y.data = response_data[1]
                response.theta.data = response_data[2]
                response.success.data = response_data[3]
                return response
            else:
                response.success.data = False
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

    def filter_points_between_planes(self, pcd, a, b, c, d):

        # 将 PointCloud 对象中的点转换为 numpy 数组
        points = np.asarray(pcd.points)

        # Filter the points based on angle (optional step)
        points_in_view = self.filter_by_angle(points, angle_threshold=30)
        num_points = len(points_in_view)
        print("视野角度内的点个数为:", num_points)

        # 计算点到地平面的距离
        height = (a * points_in_view[:, 0] + b * points_in_view[:, 1] + c * points_in_view[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        
        # 筛选出距离在 [0, max_height] 之间的点，且要求距离为正（即在平面上方）
        mask1 = (height > 0) & (height <= self.detection_max_height)
        
        points_in_view = points_in_view[mask1]
        # 计算每个点到原点的欧氏距离
        distance_to_origin = np.linalg.norm(points_in_view, axis=1)
        
        # 筛选出距离原点小于 max_distance 的点
        mask2 = (distance_to_origin <= self.detection_max_height) & (distance_to_origin >= self.detection_min_distance)
        
        # 获取同时满足两个条件的点
        selected_points = points_in_view[mask2]

        if len(selected_points) != 0:
            # 将筛选出的点转为 Open3D 的 PointCloud 对象
            selected_pcd = o3d.geometry.PointCloud()
            selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
            selected_pcd.paint_uniform_color([0, 0, 1])      # 选定点显示为蓝色
            return selected_pcd
        else:
            print("没有满足条件的点云数据!")
            return None

    def extract_ground_points(self, points, distance_threshold=0.01):

        filtered_pcd = o3d.geometry.PointCloud()
        
        # Ensure you're extracting only the 'x', 'y', and 'z' components and converting to np.float64
        points = np.asarray(points)
        if points.dtype.names is not None:
            points = points[['x', 'y', 'z']]  # Extract x , y, z fields

        points = points.view(np.float64).reshape(-1, 3)  # Flatten the data to 3 columns (x, y, z)
        filtered_pcd.points = o3d.utility.Vector3dVector(points)

        plane_model, inliers = filtered_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        ground_points_pcd = filtered_pcd.select_by_index(inliers, invert=False)
        non_ground_points_pcd = filtered_pcd.select_by_index(inliers, invert=True)

        if len(ground_points_pcd.points) != 0:
            # ground_points_pcd.paint_uniform_color([0, 1, 0])
            print("地面点数量：", len(ground_points_pcd.points))
            return ground_points_pcd, plane_model, non_ground_points_pcd
        else:
            return None, None, None

    # 计算雷达坐标系到世界坐标系的欧式变换矩阵
    def compute_inverse_transformation_matrix(self, ground_normal, radar_translation):
        """
        Compute the transformation matrix from radar coordinates to world coordinates.

        Args:
            ground_normal (list or ndarray): The ground normal vector in world coordinates.
            radar_translation (list or ndarray): The translation vector of the radar in world coordinates.

        Returns:
            ndarray: 4x4 transformation matrix from radar coordinates to world coordinates.
        """
        # Normalize ground normal vector
        n = np.array(ground_normal)
        n = n / np.linalg.norm(n)

        # Target z-axis in radar coordinates
        z_axis = np.array([0, 0, 1])

        # Compute rotation axis (cross product) and angle
        rotation_axis = np.cross(z_axis, n)  # Swap the cross order for inverse rotation
        sin_theta = np.linalg.norm(rotation_axis)
        rotation_axis = rotation_axis / (sin_theta + 1e-6)  # Normalize

        cos_theta = np.dot(z_axis, n)

        # Rodrigues' rotation formula components
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])

        R = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)

        # Inverse rotation is the transpose of the rotation matrix
        R_inv = R.T

        # Translation vector
        T = np.array(radar_translation).reshape(3, 1)

        # Compute the inverse translation
        T_inv = -R_inv @ T

        # Construct the 4x4 inverse transformation matrix
        transformation_matrix_inv = np.eye(4)
        transformation_matrix_inv[:3, :3] = R_inv
        transformation_matrix_inv[:3, 3] = T_inv.ravel()

        return transformation_matrix_inv

    def apply_transformation(self, point, transformation_matrix_inv):

        # Convert point to homogeneous coordinates
        point_homogeneous = np.append(point, 1)  # [x_r, y_r, z_r] -> [x_r, y_r, z_r, 1]

        # Apply the transformation matrix
        transformed_point_homogeneous = transformation_matrix_inv @ point_homogeneous

        # Convert back to Cartesian coordinates (normalize if necessary)
        transformed_point = transformed_point_homogeneous[:3] / transformed_point_homogeneous[3]

        return transformed_point

    def extract_front_line(self, a, b, c, d, transformation_matrix_inv, selected_points_pcd):
        """
        提取点云中的主分支，动态调整解释方差和距离权重。
        """
        # 将点云转换为 numpy 数组
        points = np.asarray(selected_points_pcd.points)

        # 高度筛选
        distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        mask = (distances > self.extract_front_line_line_min_threshold) & (distances <= self.extract_front_line_line_max_threshold)
        points = points[mask]  

        # 检查点云数据是否为空
        if points.shape[0] == 0:
            print("No points found after filtering.")
            return None, None

        print(f"Selected points shape: {points.shape}")

        #根据点云到雷达的距离动态调整聚类的参数：eps较小的时候适用于近点，较大适用于y远点
        centroid = np.mean(points, axis=0)  # 计算点云的质心
        distance_to_ridar = np.linalg.norm(centroid)  # 质心到雷达原点的距离
        eps = 0.0015 * (distance_to_ridar - 1) * (distance_to_ridar - 1) * (distance_to_ridar - 1) + 0.012
        print(f"聚类eps:{eps}")

        # 使用 DBSCAN 聚类识别点群
        clustering = DBSCAN(eps, min_samples=self.extract_front_line_min_samples).fit(points[:, :2])
        labels = clustering.labels_

        # 提取所有不同的聚类标签
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            print("未识别到任何聚类。")
            return None, None

        label_counts = Counter(labels[labels != -1])
        top_labels = [label for label, _ in label_counts.most_common(2)]

        print("识别到的聚类数:", len(unique_labels))
        print("点数最多的五个聚类标签:", top_labels)

        # 存储聚类特征
        cluster_centers = []
        original_clusters = []
        straightness_scores = []  # 用于存储每个聚类的直线性得分(第一主成分的方差解释率)
        main_directions = [] #存储每个聚类的第一主成分的方向

        for label in top_labels:
            cluster_points = points[labels == label]
            if len(cluster_points) > 0:
                original_clusters.append(cluster_points)

                # 计算聚类中心
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_centers.append(cluster_center)

                if len(cluster_points) < 2:  # 点不足以拟合直线时，跳过该聚类
                    straightness_scores.append(0)
                    continue

                # 使用 PCA 分析点云数据
                pca = PCA(n_components=2)
                pca.fit(cluster_points)

                # 第一主成分的方差解释率
                first_component_variance_ratio = pca.explained_variance_ratio_[0]
                straightness_scores.append(first_component_variance_ratio)
                main_directions.append(pca.components_[0])

                # 计算聚类之间的平均距离
                num_clusters = len(cluster_centers)
                inter_cluster_distances = []
                for i in range(num_clusters):
                    for j in range(i + 1, num_clusters):
                        distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                        inter_cluster_distances.append(distance)

        # 计算聚类之间的平均距离
        avg_inter_cluster_distance = np.mean(inter_cluster_distances) if inter_cluster_distances else 0
        print(f"聚类之间的平均距离: {avg_inter_cluster_distance:.4f}")

        # 动态调整权重
        w1 = self.increasing_function(avg_inter_cluster_distance, k = 0.2)
        w2 = 1 - w1

        # 计算综合得分
        scores = []
        for i in range(len(original_clusters)):
            straightness_score = straightness_scores[i]#聚类内点占比和得分成正相关
            print(f"聚类{i}直线性得分:{straightness_score}")
            #这里计算的距离为转换到世界坐标系之后x和y轴分量到雷达原点的距离
            real_world_target = self.apply_transformation(cluster_centers[i],transformation_matrix_inv)
            real_world_origin = self.apply_transformation([0,0,0],transformation_matrix_inv)
            target_xy = real_world_target[:2]
            origin_xy = real_world_origin[:2]
            cluster_distance_score = 1 / (np.linalg.norm(target_xy - origin_xy) + 1e-6)#聚类距离雷达原点的距离和得分成负相关
            score = w1 * cluster_distance_score + w2 * straightness_score
            scores.append(score)

        # 选择最佳聚类
        best_cluster_index = np.argmax(scores)
        best_cluster_points = original_clusters[best_cluster_index]

        print(f"最佳聚类索引: {best_cluster_index}")
        print(f"最佳聚类评分: {scores[best_cluster_index]:.4f}")
        print(f"动态权重 - 距离权重: {w1:.4f}, 直线性权重: {w2:.4f}")
        print(f"最优聚类的直线性得分: {straightness_scores[best_cluster_index]:.4f}")

        # 剔除掉直线上面那些连接的细小分支
        main_direction = main_directions[best_cluster_index]  # 第一主成分为主方向

        # 归一化主方向向量
        main_direction = main_direction / np.linalg.norm(main_direction)

        # 计算每个点到拟合直线的投影
        centroid = np.mean(best_cluster_points, axis=0)  # 点云的质心
        vec_to_points = best_cluster_points - centroid
        projections = np.dot(vec_to_points, main_direction)
        proj_points = centroid + np.outer(projections, main_direction)

        # 计算点到拟合直线的距离
        distances = np.linalg.norm(best_cluster_points - proj_points, axis=1)

        # 筛选距离拟合直线较近的点（例如距离小于0.01）
        distance_threshold = 0.03
        mask = distances < distance_threshold
        filtered_points = best_cluster_points[mask]

        # 转换为 Open3D 点云对象
        best_cluster_pcd = o3d.geometry.PointCloud()
        best_cluster_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        best_cluster_pcd.paint_uniform_color([1, 0, 0])

        # 处理其他聚类
        other_clusters_pcds = []
        for i, cluster_points in enumerate(original_clusters):
            if i != best_cluster_index:
                other_cluster_pcd = o3d.geometry.PointCloud()
                other_cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                other_cluster_pcd.paint_uniform_color([1, 1, 0])
                other_clusters_pcds.append(other_cluster_pcd)

        return best_cluster_pcd, other_clusters_pcds


    def fit_line_and_find_extremes(self, point_cloud_pcd, target_point_cloud_pcd):

        point_cloud = np.asarray(point_cloud_pcd.points)
        if point_cloud.shape[0] == 0:
            return None, None, None, None
        
        # Step 1: PCA 拟合直线
        mean_point = np.mean(point_cloud, axis=0)  # 计算聚类点的均值
        centered_points = point_cloud - mean_point
        u, s, vh = np.linalg.svd(centered_points, full_matrices=False)  # SVD 分解
        principal_direction = vh[0]  # 主方向向量
        principal_direction = principal_direction / np.linalg.norm(principal_direction)

        distances = np.dot(point_cloud - point_cloud.mean(axis=0), principal_direction)  # 每个点的投影值
        # 找到投影值最小和最大的两个点作为端点
        endpoints_indices = [np.argmin(distances), np.argmax(distances)]  # 投影值最小和最大的点
        endpoints = point_cloud[endpoints_indices]
        start_point = endpoints[0].reshape(-1)
        end_point = endpoints[1].reshape(-1)
        print("聚类之后得到线段的端点：",start_point,end_point)
        print("line_direction:",principal_direction)

        #沿直线方向在空间内扩展点云，目的是求出准确的直线两个端点
        extended_points = self.extend_points_along_line_band(point_cloud_pcd, target_point_cloud_pcd, start_point, end_point, principal_direction,
                                                             band_width= self.extract_front_line_line_max_threshold - self.extract_front_line_line_min_threshold)
        extended_pcd = o3d.geometry.PointCloud()
        extended_pcd.points = o3d.utility.Vector3dVector(extended_points)
        extended_pcd.paint_uniform_color([1, 0, 0])

        # 根据x坐标选取端点（x坐标较小的为左端点，较大的为右端点）
        left_point = extended_points[np.argmin(extended_points[:, 0])]  # x最小的点
        right_point = extended_points[np.argmax(extended_points[:, 0])]  # x最大的点
        center_point = (left_point + right_point) / 2
        
        return left_point, right_point, center_point, principal_direction, extended_pcd

    # 定义一个函数来计算点到直线带状区域的距离（向量化）
    def filter_points_within_band(self, points, line_start, line_direction, band_width):
        # 计算每个点到线起点的向量
        point_vectors = points - line_start  # (N, 3)
        
        # 计算点到直线的垂直距离
        cross_prod = np.cross(line_direction, point_vectors)  # (N, 3)
        distance_to_line = np.linalg.norm(cross_prod, axis=1) / np.linalg.norm(line_direction)
        
        # 计算点在直线方向上的投影距离
        projection = np.dot(point_vectors, line_direction)  # (N,)

        # 判断该点是否在带状区域内：
        # - 距离小于带状区域宽度
        # - 投影距离在有效扩展范围内

        return (distance_to_line < band_width) & (np.abs(projection) <= band_width)
    
    # 扩展点云：沿着拟合的直线方向扩展带状区域
    def extend_points_along_line_band(self, point_cloud_pcd, target_point_cloud_pcd, line_start, line_end, line_direction, band_width=0.05, extension_gap = 0.05, max_extension_distance=1.8):
        before_extend_points = np.asarray(point_cloud_pcd.points)
        extended_points = before_extend_points.tolist()  # 先将前端聚类的点加入扩展点云

        points = np.asarray(target_point_cloud_pcd.points)
        
        # 扩展方向：从前端横梁点云的起始点（line_start）和终止点（line_end）分别向外扩展
        for direction in [1, -1]:  # 1 为向正方向扩展，-1 为向负方向扩展
            print("开始根据原始点云扩展前端检测横梁...")
            current_line_point = line_start if direction == 1 else line_end
            current_line_direction = line_direction if direction == 1 else -line_direction
            extension_distance = 0  # 用于控制扩展的距离

            # 扩展过程
            while extension_distance < max_extension_distance:  # 设置扩展的最大距离
                extension_distance += extension_gap  # 每次扩展带状区域宽度

                # 计算点到直线带状区域的距离（向量化）
                filtered_points_within_band = self.filter_points_within_band(
                    points, current_line_point, current_line_direction, band_width
                )
                
                # 找到带状区域内的点
                if np.any(filtered_points_within_band):  # 如果有点在带状区域内
                    # 将所有符合条件的点加入扩展点云
                    extended_points.extend(points[filtered_points_within_band].tolist())
                    # 更新扩展区域的起始点或终止点
                    current_line_point += extension_gap * current_line_direction  # 更新扩展点的位置
                else:
                    break  # 如果在扩展的距离内没有找到符合条件的点，停止扩展
        return np.array(extended_points)
    
    def detect_pallet_feet_with_orientation(self, point_cloud, a,b,c,d,radar_origin=(0, 0, 0), eps=0.05, min_samples=8, max_distance=2.5,line_threshold = 0.3):
        """
        检测托盘脚点云簇，并计算托盘最前方中心立柱的中心点坐标和托盘朝向。

        参数：
        - point_cloud: 输入点云数据 (Open3D 点云对象)
        - radar_origin: 雷达位置 (默认是原点)
        - eps: DBSCAN 中的距离阈值
        - min_samples: DBSCAN 中的最小样本数
        - max_distance: 从雷达到点的最大距离 (单位: 米)

        返回：
        - feet_clouds: 一个包含托盘脚点云的列表，每个点云为 Open3D 点云对象。
        - center_column_center: 托盘最前方中心立柱的中心点坐标 (numpy.ndarray)
        - pallet_orientation: 托盘的朝向向量 (numpy.ndarray)
        """
        # 将点云转换为 numpy 数组
        points = np.asarray(point_cloud.points)

        # 检查点云是否为空
        if points.shape[0] == 0:
            print("点云为空！")
            return [], None, None

        # 按距离过滤，只保留离雷达原点 max_distance 范围内的点
        distances_to_origin = np.linalg.norm(points - np.array(radar_origin), axis=1)
        points = points[distances_to_origin <= max_distance]

        # DBSCAN 聚类分析
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        unique_labels = set(labels)

        # 计算每个簇的点数，并保留点数最多的前 5 个簇
        cluster_sizes = {label: (labels == label).sum() for label in unique_labels if label != -1}
        top_labels = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)[:5]

        # 对每个点数最多的簇进行竖直性判断
        feet_clusters = []
        for label in top_labels:
            cluster_points = points[labels == label]

            # 检查点云簇是否满足竖直性
            cov_matrix = np.cov(cluster_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            vertical_direction = eigenvectors[:, 0]  # 最小特征值对应的方向
            if np.abs(vertical_direction[2]) > 0.9:  # 接近竖直方向
                feet_clusters.append(cluster_points)

        # 检查是否有足够的簇
        if len(feet_clusters) == 0:
            print("未检测到任何竖直方向的点云簇！")
            return [], None, None

        # 按簇到雷达原点的距离排序
        feet_clusters = sorted(
            feet_clusters,
            key=lambda x: np.linalg.norm(np.mean(x, axis=0) - np.array(radar_origin))
        )

        # 计算最前方中心立柱的中心点
        center_column = feet_clusters[0]  # 最前方的簇

        center_column_center = np.mean(center_column, axis=0)  # 计算质心

        #只保留和中心点y轴坐标之差小于阈值的点
        mask = ((center_column[:,1] - center_column_center[1]) <= 0) & ((center_column[:,1] - center_column_center[1]) >= -0.06)

        # 过滤点云
        center_column = center_column[mask]
        print("过滤之后点云的数量：",len(center_column))
        # 过滤 NaN 和 Inf 值
        center_column = center_column[~np.isnan(center_column).any(axis=1)]
        center_column = center_column[~np.isinf(center_column).any(axis=1)]

        #提取前部平面点云上方的横梁
        #高度筛选
        distances = (a * center_column[:, 0] + b * center_column[:, 1] + c * center_column[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        # mask = (distances > line_threshold - 0.03) & (distances < line_threshold + 0.02)
        mask = (distances > (line_threshold - 0.18)) & (distances <= line_threshold)
        front_line_column = center_column[mask]
        front_line_column_pcd = o3d.geometry.PointCloud()
        front_line_column_pcd.points = o3d.utility.Vector3dVector(front_line_column)
        front_line_column_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 设置为红色

        #然后将过滤后的点云拟合成一条直线，并求出直线的方向向量
        midpoint, principal_direction = self.fit_line_with_outlier_removal(front_line_column_pcd.points)

        return front_line_column_pcd, center_column_center, principal_direction

    def fit_line_with_outlier_removal(self, points, k=5, noise_threshold=0.05, distance_threshold=0.05):

        # 将 points 转换为 NumPy 数组
        points = np.asarray(points)
        # Step 1: 去除噪声
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, _ = nbrs.kneighbors(points)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        filtered_points = points[mean_distances < noise_threshold]

        # Step 2: 初步拟合直线
        pca = PCA(n_components=1)
        pca.fit(filtered_points)
        line_direction = pca.components_[0]
        centroid = np.mean(filtered_points, axis=0)

        # Step 3: 剔除离群点
        vector_to_point = filtered_points - centroid
        projections = np.dot(vector_to_point, line_direction)
        projected_points = centroid + np.outer(projections, line_direction)
        distances = np.linalg.norm(filtered_points - projected_points, axis=1)
        refined_points = filtered_points[distances < distance_threshold]

        # Step 4: 最终拟合
        pca.fit(refined_points)
        final_direction = pca.components_[0] * -1
        final_centroid = np.mean(refined_points, axis=0)

        return final_centroid, final_direction

    def save_point_cloud_as_ply(self, points, filename):
        """将点云数据保存为 PLY 文件"""
        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()

        # 将 NumPy 数组转换为 Open3D 点云对象
        pcd.points = o3d.utility.Vector3dVector(points)

        # 保存点云为 PLY 文件
        o3d.io.write_point_cloud(filename, pcd)
        self.get_logger().info(f'Point cloud saved as {filename}')
    
def main(args = None):
    print("Start Registration Server")
    rclpy.init()
    node = RegistrationServer()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
