import time
import pcl
import pcl.pcl_visualization
from arun_lib.data_loader import FileLocation, FrameDataLoader
from arun_lib.visualization.visual_3d import Visualization3D
# 使用DBSCAN来进行去除孤立点。
from sklearn.cluster import DBSCAN
import numpy as np

if __name__ == '__main__':
    # 定位数据集的位置
    root_path = "../../arun_data/data_set"
    output_path = "70_radar5_lvbo"
    data_locations = FileLocation(root_dir=root_path, output_dir=output_path,radar_type='radar_5frame')
    # 装载一帧数据
    frame_data = FrameDataLoader(file_locations=data_locations, frame_name="03047")
    # 取出radar的原始数据
    radar_raw = frame_data.radar_data.reshape(-1,7)
    radar_points = radar_raw[:,:3]
    print(radar_points.shape)
    # 对radar的点做滤波处理
    mask =(radar_points[:,2]>=0)&(np.abs(radar_points[:,0])<=70)&(np.abs(radar_points[:,1]<=70))
    filter_radar_points = radar_points[mask]
    print(filter_radar_points.shape)
    # 使用DBSCAN来对孤立点聚类,eps为邻居的最大距离，min_samples是最小邻居数
    s = time.time()
    db = DBSCAN(eps=1, min_samples=5)
    gudian = db.fit_predict(filter_radar_points)
    gudian_mask = (gudian!=-1)
    filter_radar_points = filter_radar_points[gudian_mask]
    print(filter_radar_points.shape)
    print('time:', time.time() - s)
    # 将处理后的radar数据重新装载回去
    frame_data.set_radar_scan(filter_radar_points)
    # 开始欧氏距离分类的处理时间
    s = time.time()
    radar_cloud = pcl.PointCloud(filter_radar_points)
    tree = radar_cloud.make_kdtree()
    eu_clu = radar_cloud.make_EuclideanClusterExtraction()
    # 三个参数设置
    eu_clu.set_ClusterTolerance(0.8)
    eu_clu.set_MinClusterSize(5)
    eu_clu.set_MaxClusterSize(100)
    eu_clu.set_SearchMethod(tree)
    # 得到分类结果
    cluster_indices = eu_clu.Extract()
    print('time:', time.time() - s)
    print('cluster_indices : ' + str(len(cluster_indices)) + "  .")
    # # 可视化被处理的点
    vis_3d = Visualization3D(frame_data)
    vis_3d.draw_plot(radar_origin_plot=True,
                 lidar_origin_plot=True,
                 camera_origin_plot=True,
                 lidar_points_plot=True,
                 radar_color_points_plot=True,
                 cluster_indices=cluster_indices,
                 annotations_plot=False,
                 own_annotations_plot=True,
                 grid_visible=True,
                 write_to_html=True,
                 html_name=output_path)


