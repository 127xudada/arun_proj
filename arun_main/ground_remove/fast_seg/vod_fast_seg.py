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




if __name__ == '__main__':
    # 定位数据集的位置
    root_path = "../../../arun_data/data_set"
    output_path = "3d_lidar_seg"
    output_path_groud = "3d_lidar_seg_groud"
    data_locations = FileLocation(root_dir=root_path, output_dir=output_path, radar_type='radar_5frame')
    print(data_locations.lidar_dir)
    # 装载一帧数据
    frame_data = FrameDataLoader(file_locations=data_locations, frame_name="01047")
    frame_data_non_ground = frame_data
    frame_data_groud = frame_data
    # 取出lidar的原始数据
    lidar_raw = frame_data.lidar_data
    lidar_points = lidar_raw[:,:3]
    print(lidar_points.shape)
    # 开始计算地面分割的处理时间
    s = time.time()
    # 翻转Z轴
    lidar_points = lidar_points * np.array([1, 1, -1])  # revert the z axis
    # 从点云中分割除出地面 Fast_segment的核心实现代码
    process = ground_removal.Processor(n_segments=140, n_bins=160, line_search_angle=0.3, max_dist_to_line=0.15,
                                       sensor_height=1.73, max_start_height=0.5, long_threshold=8)
    # 调用process接口返回地面和非地面的点云数据
    vel_non_ground, vel_ground = process(lidar_points)
    # 翻转Z轴
    vel_non_ground = vel_non_ground * np.array([1, 1, -1])  # revert the z axis
    frame_data_non_ground.set_lidar_scan(vel_non_ground)
    # frame_data_groud.set_lidar_scan(vel_ground)
    print("去除地面点云的大小:{}".format(vel_non_ground.shape))
    print("地面点云的大小:{}".format(vel_ground.shape))
    # 可视化去除地面的点云
    vis_3d = Visualization3D(frame_data_non_ground)
    vis_3d.draw_plot(radar_origin_plot=True,
                     lidar_origin_plot=True,
                     camera_origin_plot=True,
                     lidar_points_plot=True,
                     radar_color_points_plot=False,
                     cluster_indices=None,
                     annotations_plot=True,
                     own_annotations_plot=False,
                     grid_visible=True,
                     write_to_html=True,
                     html_name=output_path)
    # # # 可视化地面点云
    # vis_3d_groud = Visualization3D(frame_data_groud)
    # vis_3d_groud.draw_plot(radar_origin_plot=True,
    #                  lidar_origin_plot=True,
    #                  camera_origin_plot=True,
    #                  lidar_points_plot=True,
    #                  radar_color_points_plot=False,
    #                  cluster_indices=None,
    #                  annotations_plot=False,
    #                  own_annotations_plot=False,
    #                  grid_visible=True,
    #                  write_to_html=True,
    #                  html_name=output_path_groud)
    # # 可视化原始地面点云
    vis_3d_groud = Visualization3D(frame_data_groud)
    vis_3d_groud.draw_plot(radar_origin_plot=True,
                           lidar_origin_plot=True,
                           camera_origin_plot=True,
                           lidar_points_plot=True,
                           radar_color_points_plot=False,
                           cluster_indices=None,
                           annotations_plot=False,
                           own_annotations_plot=False,
                           grid_visible=True,
                           write_to_html=True,
                           html_name=output_path_groud)
