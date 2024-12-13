import open3d as o3d
import numpy as np
from mise.pointcloud import estimate_normal

def visualization(src_pcd, tgt_pcd, pred_trans):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    key_to_callback = {ord("K"): change_background_to_black}
    if not src_pcd.has_normals():
        estimate_normal(src_pcd)
        estimate_normal(tgt_pcd)
    src_pcd.paint_uniform_color([1, 0.706, 0])
    src_box = src_pcd.get_oriented_bounding_box()
    src_box.color = (0 ,1, 0)
    tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
    tgt_box = tgt_pcd.get_oriented_bounding_box()
    tgt_box.color = (0, 1, 0)

    o3d.visualization.draw_geometries_with_key_callbacks([src_pcd, tgt_pcd, src_box, tgt_box], key_to_callback)
    src_pcd.transform(pred_trans)
    src_box = src_pcd.get_oriented_bounding_box()
    src_box.color = (1, 0, 0)
    o3d.visualization.draw_geometries_with_key_callbacks([src_pcd, tgt_pcd, src_box, tgt_box], key_to_callback)

def preprocess_point_cloud(pcd, voxel_size):
    if pcd is None:
        print("未读取到点云数据")
        return 
    """
    点云预处理：下采样、法线估计和 FPFH 特征计算
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    # 计算法向量
    radius_normal = voxel_size * 2
    if not pcd_down.has_normals():
        print("点云缺少法向量信息，正在计算法向量...")
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, 
                max_nn=30
            )
        )
        print("法向量计算完成")
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    使用 RANSAC + 4PCS 进行基于特征的粗配准
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,  # 4PCS 的 4 点约束
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def refine_registration(source, target, init_transformation, voxel_size):
    """
    使用 ICP 进行精配准
    """
    radius_normal = voxel_size * 2
    if not target.has_normals():
        print("点云缺少法向量信息，正在计算法向量...")
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, 
                max_nn=30
            )
        )
        print("法向量计算完成")
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result

def main():
    # 加载源点云和目标点云
    source_pcd = o3d.io.read_point_cloud("/home/yuyifan/point_cloud/output/two_target.pcd")
    target_pcd = o3d.io.read_point_cloud("/home/yuyifan/point_cloud/output/two/0010.pcd")

    # 定义体素大小，用于下采样
    voxel_size = 0.05

    # 点云预处理
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

    # 基于特征的粗配准
    print("开始基于特征的粗配准...")
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("粗配准结果:")
    print(result_ransac)

    # 精配准：使用 ICP
    print("开始ICP精配准...")
    result_icp = refine_registration(source_pcd, target_pcd, result_ransac.transformation, voxel_size)
    print("精配准结果:")
    print(result_icp)
    
    # 变换源点云并显示结果
    final_trans = source_pcd.transform(result_icp.transformation)
    visualization(source_pcd, target_pcd, final_trans)

if __name__ == "__main__":
    main()
