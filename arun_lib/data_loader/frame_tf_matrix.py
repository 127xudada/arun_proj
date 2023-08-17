import json
import os
import logging
from typing import Optional
import numpy as np
from arun_lib.data_loader.frame_load import FrameDataLoader
from arun_lib.data_loader.file_location import FileLocation

class FrameTransformMatrix:
    """
    本类负责提供可能的坐标系之间的齐次变换矩阵
    """
    def __init__(self, fram_data_loader: FrameDataLoader):
        # 创建数据装载器
        self.frame_data_loader: FrameDataLoader = fram_data_loader
        """各坐标系本地变换"""
        # 相机投影矩阵
        self._camera_projection_matrix: Optional[np.ndarray] = None
        # 相机到激光雷达的变换矩阵
        self._T_camera_lidar: Optional[np.ndarray] = None
        # 相机到雷达的变换矩阵
        self._T_camera_radar: Optional[np.ndarray] = None
        # 激光雷达到相机的变换矩阵
        self._T_lidar_camera: Optional[np.ndarray] = None
        # 雷达到相机的变换矩阵
        self._T_radar_camera: Optional[np.ndarray] = None
        # 激光雷达到雷达的变换矩阵
        self._T_lidar_radar: Optional[np.ndarray] = None
        # 雷达到激光雷达的变换矩阵
        self._T_radar_lidar: Optional[np.ndarray] = None
        """各坐标系全局变换"""
        # 里程计到相机的变换矩阵
        self._T_odom_camera: Optional[np.ndarray] = None
        # 地图到相机的变换矩阵
        self._T_map_camera: Optional[np.ndarray] = None
        # UTM坐标系到相机的变换矩阵
        self._T_UTM_camera: Optional[np.ndarray] = None
        # 相机到里程计的变换矩阵
        self._T_camera_odom: Optional[np.ndarray] = None
        # 相机到地图的变换矩阵
        self._T_camera_map: Optional[np.ndarray] = None
        # 相机到UTM坐标系的变换矩阵
        self._T_camera_UTM: Optional[np.ndarray] = None

    @property
    def camera_projection_matrix(self):
        """
        获取相机投影矩阵
        :return: Numpy array of the camera projection matrix.
        """
        if self._camera_projection_matrix is not None:
            # When the data is already loaded.
            return self._camera_projection_matrix
        else:
            # Load data if it is not loaded yet.
            self._camera_projection_matrix, self._T_camera_lidar = self.get_sensor_transforms('lidar')
            return self._camera_projection_matrix

    @property
    def t_camera_lidar(self):
        """
        返回从激光雷达坐标系到相机坐标系的齐次变换矩阵
        :return: Numpy array of the homogeneous transform matrix from the lidar frame, to the camera frame.
        """
        if self._T_camera_lidar is not None:
            # When the data is already loaded.
            return self._T_camera_lidar
        else:
            # Load data if it is not loaded yet.
            self._camera_projection_matrix, self._T_camera_lidar = self.get_sensor_transforms('lidar')
            return self._T_camera_lidar

    @property
    def t_camera_radar(self):
        """
        返回从雷达坐标系到相机坐标系的齐次变换矩阵
        :return: Numpy array of the homogeneous transform matrix from the radar frame, to the camera frame.
        """
        if self._T_camera_radar is not None:
            # When the data is already loaded.
            return self._T_camera_radar
        else:
            # Load data if it is not loaded yet.
            self._camera_projection_matrix, self._T_camera_radar = self.get_sensor_transforms('radar')
            return self._T_camera_radar

    @property
    def t_lidar_camera(self):
        """
        :return: Numpy array of the homogeneous transform matrix from the camera frame, to the lidar frame.
        """
        if self._T_lidar_camera is not None:
            # When the data is already loaded.
            return self._T_lidar_camera
        else:
            # Calculate data if it is calculated yet.
            self._T_lidar_camera = np.linalg.inv(self.t_camera_lidar)
            return self._T_lidar_camera

    @property
    def t_radar_camera(self):
        """
        :return: Numpy array of the homogeneous transform matrix from the camera frame, to the radar frame.
        """
        if self._T_radar_camera is not None:
            # When the data is already loaded.
            return self._T_radar_camera
        else:
            # Calculate data if it is calculated yet.
            self._T_radar_camera = np.linalg.inv(self.t_camera_radar)
            return self._T_radar_camera

    @property
    def t_lidar_radar(self):
        """
        :return: Numpy array of the homogeneous transform matrix from the radar frame, to the lidar frame.
        """
        if self._T_lidar_radar is not None:
            # When the data is already loaded.
            return self._T_lidar_radar
        else:
            # Calculate data if it is calculated yet.
            self._T_lidar_radar = np.dot(self.t_lidar_camera, self.t_camera_radar)
            return self._T_lidar_radar

    @property
    def t_radar_lidar(self):
        """
        :return: Numpy array of the homogeneous transform matrix from the lidar frame, to the radar frame.
        """
        if self._T_radar_lidar is not None:
            # When the data is already loaded.
            return self._T_radar_lidar
        else:
            # Calculate data if it is calculated yet.
            self._T_radar_lidar = np.dot(self.t_radar_camera, self.t_camera_lidar)
            return self._T_radar_lidar

    @property
    def t_odom_camera(self):
        """
        :return: Numpy array of the homogeneous transform matrix from the camera frame, to the odom frame.
        """
        if self._T_odom_camera is not None:
            # When the data is already loaded.
            return self._T_odom_camera
        else:
            # Load data if it is not loaded yet.
            self._T_odom_camera, self._T_map_camera, self._T_UTM_camera = self.get_world_transform()
            return self._T_odom_camera

    @property
    def t_map_camera(self):
        """
        :return: Numpy array of the homogeneous transform matrix from the camera frame, to the map frame.
        """
        if self._T_map_camera is not None:
            # When the data is already loaded.
            return self._T_map_camera
        else:
            # Load data if it is not loaded yet.
            self._T_odom_camera, self._T_map_camera, self._T_UTM_camera = self.get_world_transform()
            return self._T_map_camera

    @property
    def t_utm_camera(self):
        """
        :return: Numpy array of the homogeneous transform matrix from the camera frame, to the UTM frame.
        """
        if self._T_UTM_camera is not None:
            # When the data is already loaded.
            return self._T_UTM_camera
        else:
            # Load data if it is not loaded yet.
            self._T_odom_camera, self._T_map_camera, self._T_UTM_camera = self.get_world_transform()
            return self._T_UTM_camera

    @property
    def t_camera_odom(self):
        """
        :return: Numpy array of the homogeneous transform matrix from the odom frame, to the camera frame.
        """
        if self._T_camera_odom is not None:
            # When the data is already loaded.
            return self._T_camera_odom
        else:
            # Calculate data if it is calculated yet.
            self._T_camera_odom = np.linalg.inv(self.t_odom_camera)
            return self._T_camera_odom

    @property
    def t_camera_map(self):
        """
        :return: Numpy array of the homogeneous transform matrix from the map frame, to the camera frame.
        """
        if self._T_camera_map is not None:
            # When the data is already loaded.
            return self._T_camera_map
        else:
            # Calculate data if it is calculated yet.
            self._T_camera_map = np.linalg.inv(self.t_map_camera)
            return self._T_camera_map

    @property
    def t_camera_utm(self):
        """
        :return: Numpy array of the homogeneous transform matrix from the UTM frame, to the camera frame.
        """
        if self._T_camera_UTM is not None:
            # When the data is already loaded.
            return self._T_camera_UTM
        else:
            # Calculate data if it is calculated yet.
            self._T_camera_UTM = np.linalg.inv(self.t_utm_camera)
            return self._T_camera_UTM

    def get_sensor_transforms(self, sensor: str):  # -> Optional[(np.ndarray, np.ndarray)]:
        """
        根据传感器的类型，从数据集中获取相应的内部和外部变换矩阵 :radar 或者 lidar
        :param sensor: Sensor name in string for which the transforms to be read from the dataset.
        :return: A numpy array tuple of the intrinsic, extrinsic transform matrix.
        """
        # 获取radar的变换矩阵
        if sensor == 'radar':
            try:
                # 从radar_calib_dir从雷达校准数据中读出变换矩阵
                calibration_file = os.path.join(self.frame_data_loader.data_loactions.radar_calib_dir,
                                                f'{self.frame_data_loader.frame_name}.txt')
            except FileNotFoundError:
                logging.error(f"{self.frame_data_loader.frame_name}.txt does not exist at"
                              f" location: {self.frame_data_loader.data_loactions.radar_calib_dir}!")
                return None, None
        # 获取lidar的变换矩阵
        elif sensor == 'lidar':
            try:
                # 从lidar_calib_dir从雷达校准数据中读出变换矩阵
                calibration_file = os.path.join(self.frame_data_loader.data_loactions.lidar_calib_dir,
                                                f'{self.frame_data_loader.frame_name}.txt')
            except FileNotFoundError:
                logging.error(f"{self.frame_data_loader.frame_name}.txt does not exist at"
                              f" location: {self.frame_data_loader.data_loactions.lidar_calib_dir}!")
                return None, None
        else:
            raise AttributeError('Not valid sensor')
        # 从calibration_file中读出对应的内部和外部变换矩阵
        with open(calibration_file, "r") as f:
            lines = f.readlines()
            intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
            extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
            extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)
        return intrinsic, extrinsic

    def get_world_transform(self):  # -> Optional[(np.ndarray, np.ndarray, np.ndarray)]:
        """
        从数据集中获取世界坐标系变换矩阵
        :return: A numpy array tuple of the t_odom_camera, t_map_camera and t_utm_camera matrices.
        """
        try:
            pose_file = os.path.join(self.frame_data_loader.data_loactions.pose_dir,
                                     f'{self.frame_data_loader.frame_name}.json')
        except FileNotFoundError:
            logging.error(f"{self.frame_data_loader.data_loactions}.json does not exist at"
                          f" location: {self.frame_data_loader.data_loactions.pose_dir}!")
            return None, None, None
        jsons = []
        for line in open(pose_file, 'r'):
            jsons.append(json.loads(line))
        # 从pose_file中读出位姿数据
        t_odom_camera = np.array(jsons[0]["odomToCamera"], dtype=np.float32).reshape(4, 4)
        t_map_camera = np.array(jsons[1]["mapToCamera"], dtype=np.float32).reshape(4, 4)
        t_utm_camera = np.array(jsons[2]["UTMToCamera"], dtype=np.float32).reshape(4, 4)
        return t_odom_camera, t_map_camera, t_utm_camera

def homogeneous_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    对给定的点集应用齐次变换
    :param points: Points to be transformed in a Nx4 numpy array.
    :param transform: 4x4 transformation matrix in a numpy array.
    :return: Transformed points of shape Nx4 in a numpy array.
    """
    if transform.shape != (4, 4):
        raise ValueError(f"{transform.shape} must be 4x4!")
    if points.shape[1] != 4:
        raise ValueError(f"{points.shape[1]} must be Nx4!")
    # 将points数组的转置进行点乘运算，并将结果进行转置
    # 函数返回变换后的点集，形状为Nx4的numpy数组
    return transform.dot(points.T).T

def homogeneous_coordinates(points: np.ndarray) -> np.ndarray:
    """
    将给定点数组转换为齐次坐标的函数
    齐次坐标可以更方便的矩阵运算，如平移、旋转和投影等。
    齐次坐标可以使用矩阵乘法来表示复杂的几何变换，并简化了坐标变换的表示和计算
    :param points: Input ndarray of shape Nx3.
    :return: Output ndarray of shape Nx4.
    """
    if points.shape[1] != 3:
        raise ValueError(f"{points.shape[1]} must be Nx3!")

    return np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))

def project_3d_to_2d(points: np.ndarray, projection_matrix: np.ndarray):
    """
    将输入的三维数组投影到二维数组，给定一个投影矩阵。它接受两个参数：points和projection_matrix
    :param points: 齐次坐标的数组，表示待投影的三维点，数组的形状可以是任意维度，但最后一个维度必须是4，以表示齐次坐标
    :param projection_matrix: 4x4的投影矩阵，用于将三维点映射到二维平面。投影矩阵定义了从三维空间到二维平面的投影规则，可以包含透视投影、平行投影等.
    :return: 2d ndarray of the projected points.
    """
    if points.shape[-1] != 4:
        raise ValueError(f"{points.shape[-1]} must be 4!")
    # 投影矩阵与点数组的转置进行矩阵乘法，得到一个新的数组uvw，其中每个元素包含了投影后的二维坐标
    uvw = projection_matrix.dot(points.T)
    # 对uvw进行归一化处理，将每个坐标的第三个分量除以其自身，以获得齐次坐标的正规化表示
    uvw /= uvw[2]
    # 提取出前两个分量，得到一个新的数组uvs，表示投影后的二维坐标
    uvs = uvw[:2].T
    # 数将uvs数组中的坐标值进行四舍五入并转换为整数类型，以确保坐标是整数像素位置。
    uvs = np.round(uvs).astype(np.int32)
    # 函数返回uvs数组，投影后的二维点的表示
    return uvs

def canvas_crop(points, image_size, points_depth=None):
    """
    过滤位于给定帧大小之外的点
    根据给定的帧大小，筛选出位于该范围内的点，并可选择性地过滤深度小于零的点
    对点云数据或其他坐标数据进行裁剪操作，只保留位于指定帧范围内且满足深度条件的点。
    去除无效区域或筛选感兴趣的区域。
    :param points: 待过滤的点数组.
    :param image_size: 帧的大小，通常表示为(height, width)
    :param points_depth: 点的深度信息，可以是None Filters also depths smaller than 0.
    :return: Filtered points.
    """
    # 创建一个布尔索引数组idx，初始为全True
    # 点的x坐标大于0。
    idx = points[:, 0] > 0
    # 逻辑与操作(np.logical_and)，将idx与每个条件逐个相与，以筛选出符合条件的点：
    # 点的x坐标小于帧的宽度
    idx = np.logical_and(idx, points[:, 0] < image_size[1])
    # 点的y坐标大于0。
    idx = np.logical_and(idx, points[:, 1] > 0)
    # 点的y坐标小于帧的高度
    idx = np.logical_and(idx, points[:, 1] < image_size[0])
    if points_depth is not None:
        # 过滤深度小于零的点
        idx = np.logical_and(idx, points_depth > 0)
    return idx

def min_max_filter(points, max_value, min_value):
    """
    基于最大值和最小值对点进行过滤
    :param points: Points to be filtered.
    :param max_value: Maximum value.
    :param min_value: Minimum value.
    :return: Filtered points.
    """
    # 创建一个布尔索引数组idx，初始为全True
    idx = points < max_value
    # 使用逻辑与操作(np.logical_and)，将idx与每个条件逐个相与，以筛选出满足条件的点：
    idx = np.logical_and(idx, points > min_value)
    # 返回过滤后的索引数组idx，该数组对应于原始点数组中满足最大值和最小值条件的点
    return idx

def project_pcl_to_image(point_cloud, t_camera_pcl, camera_projection_matrix, image_shape):
    """
    将点云投影到相机图像帧上
    :param point_cloud: 待投影的点云数据.
    :param t_camera_pcl: 从点云坐标系到相机坐标系的变换矩阵.
    :param camera_projection_matrix: 相机的投影矩阵，大小为4x4.
    :param image_shape: 是相机图像的尺寸.
    :return: Projected points, and the depth of each point.
    """
    # 将点云的坐标转换为齐次坐标，通过在点云的每个点的末尾添加一个值为1的列，形成大小为Nx4的数组
    point_homo = np.hstack((point_cloud[:, :3],
                            np.ones((point_cloud.shape[0], 1),
                                    dtype=np.float32)))
    # 使用变换矩阵t_camera_pcl将点云从点云坐标系转换到相机坐标系，得到在相机坐标系下的点云坐标数组radar_points_camera_frame
    radar_points_camera_frame = homogeneous_transformation(point_homo,
                                                           transform=t_camera_pcl)
    # 提取每个点的深度值，存储在数组point_depth中
    point_depth = radar_points_camera_frame[:, 2]
    # 将在相机坐标系下的点云投影到相机图像平面上，得到点在图像上的像素坐标数组uvs
    uvs = project_3d_to_2d(points=radar_points_camera_frame,
                           projection_matrix=camera_projection_matrix)
    # 使用canvas_crop函数筛选出位于图像范围内的点
    filtered_idx = canvas_crop(points=uvs,
                               image_size=image_shape,
                               points_depth=point_depth)
    # 根据索引数组filtered_idx对uvs和point_depth进行过滤操作
    uvs = uvs[filtered_idx]
    point_depth = point_depth[filtered_idx]
    # 返回经过投影和过滤后的点在图像上的像素坐标数组uvs，以及每个点的深度值point_depth
    return uvs, point_depth

def transform_pcl(points: np.ndarray, transform_matrix: np.ndarray):
    """
    使用变换矩阵对齐次点进行坐标变换
    对给定的点坐标数据进行齐次变换，即将点从一个坐标系转换到另一个坐标系。
    通过调用transform_pcl函数，可以实现点云数据在不同坐标系之间的转换，
    例如将点云从一个坐标系转换到相机坐标系或世界坐标系。
    :param points: 待变换的点的坐标数据.
    :param transform_matrix: H齐次变换矩阵.
    :return: Transformed homogenous points.
    """
    # 将点的坐标转换为齐次坐标，通过在每个点的末尾添加一个值为1的列，形成大小为Nx4的数组
    point_homo = np.hstack((points[:, :3],
                            np.ones((points.shape[0], 1),
                                    dtype=np.float32)))
    # 将齐次点按照给定的变换矩阵transform_matrix进行坐标变换，得到在新坐标系下的点坐标数组
    points_new_frame = homogeneous_transformation(point_homo, transform=transform_matrix)
    # 返回经过变换后的齐次点坐标数组
    return points_new_frame


if __name__ == '__main__':
    # 数据集根目录
    root_path = "../../arun_data/data_set"
    # 找到数据位置
    data_locations = FileLocation(root_dir=root_path)
    # 将数据装载入frame_data
    frame_data = FrameDataLoader(file_locations=data_locations, frame_name="01047")
    # 验证本模块的矩阵变换获取函数
    t_mat_frame = FrameTransformMatrix(frame_data)
    print(t_mat_frame.t_camera_lidar)
    print(t_mat_frame.t_camera_odom)

