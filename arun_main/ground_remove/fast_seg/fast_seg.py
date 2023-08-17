import cv2
import time
import os
import pcl
import numpy as np
import pcl.pcl_visualization
# 地面分割库
from arun_lib.data_loader.gen_cloud import Gencloud
from arun_lib.ground_remove.Fast_Seg import lidar_projection
from arun_lib.ground_remove.Fast_Seg import ground_removal
# 点云可视化库
from arun_lib.data_loader import FileLocation, FrameDataLoader
from arun_lib.visualization.visual_3d import Visualization3D
# 可视化尝试
from mpl_toolkits.mplot3d import Axes3D
"""----------------------------------------------------------------"""
# 设置print的打印精度和是否使用科学计数法
np.set_printoptions(precision=3, suppress=True)
# 创建could生成器
Gen_cloud = Gencloud()
# 装载PCD点云数据
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..','..','..','arun_data', 'data_test', 'ref_cloud')
raw_data = np.asarray(pcl.load(os.path.join(data_dir, 'ref_cloud_09281250-168.17.ply')))
print("原始点云的大小:{}".format(raw_data.shape))
# 过滤掉激光雷达以上的点云
vel_no_top = np.asarray([vel_ms for vel_ms in raw_data if vel_ms[2] < 0])
print("过滤顶部点云的大小:{}".format(vel_no_top.shape))
# 翻转Z轴 ？？
vel_no_top = vel_no_top * np.array([1, 1, -1])  # revert the z axis
# 从点云中分割除出地面 Fast_segment的核心实现代码
# process = ground_removal.Processor(n_segments=60, n_bins=80, line_search_angle=0.2, max_dist_to_line=0.25,
#                     sensor_height=1.73, max_start_height=0.4, long_threshold=3)
process = ground_removal.Processor(n_segments=70, n_bins=80, line_search_angle=0.3, max_dist_to_line=0.15,
                    sensor_height=3.4, max_start_height=0.5, long_threshold=8)
# 调用process接口返回地面和非地面的点云数据
vel_non_ground, vel_ground = process(vel_no_top)
print("去除地面点云的大小:{}".format(vel_non_ground.shape))
print("地面点云的大小:{}".format(vel_ground.shape))
# 将地面点云与非地面点云转成pcd方便查看
Gen_cloud.points2pcd(vel_non_ground, "ref_cloud_09281250-168_non_ground.pcd")
Gen_cloud.points2pcd(vel_ground,"ref_cloud_09281250-168_ground.pcd")
# Generate BEV image 生成鸟瞰图
img_raw = lidar_projection.birds_eye_point_cloud(raw_data,
                                                 side_range=(-50, 50), fwd_range=(-50, 50),
                                                 res=0.25, min_height=-2, max_height=4)
cv2.imwrite('ref_cloud_09281250-168_raw.png', img_raw)

img_non_ground = lidar_projection.birds_eye_point_cloud(vel_non_ground,
                                                        side_range=(-50, 50), fwd_range=(-50, 50),
                                                        res=0.25, min_height=-2, max_height=4)
cv2.imwrite('ref_cloud_09281250-168_no_ground.png', img_non_ground)

img_ground = lidar_projection.birds_eye_point_cloud(vel_ground,
                                                    side_range=(-50, 50), fwd_range=(-50, 50),
                                                    res=0.25, min_height=-2, max_height=4)
cv2.imwrite('ref_cloud_09281250-168_ground.png', img_ground)
