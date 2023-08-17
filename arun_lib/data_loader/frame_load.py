import os
import numpy as np
from matplotlib import pyplot as plt
from arun_lib.data_loader.file_location import FileLocation
# 用于类型提示的工具,明确指定变量、函数参数和返回值的类型信息
from typing import Optional, List
import logging

class FrameDataLoader:
    """
        本类负责从数据集中加载单个帧的所有数据
    """
    def __init__(self,
                 file_locations: FileLocation,
                 frame_name: str):
        """
        构造framedataloader类
        :param file_locations: 指示数据的位置
        :param frame_name: 指示具体数据帧的名称
        """
        # 初始化参数
        self.data_loactions: FileLocation = file_locations
        self.frame_name: str = frame_name
        self.file_id: str = str(self.frame_name).zfill(5)
        # 装载图片、雷达、激光数据等的存储
        # Optional[np.ndarray] 表示属性可以存储 np.ndarray 类型的数据，或者为空 (None)
        self._image: Optional[np.ndarray] = None
        self._lidar_data: Optional[np.ndarray] = None
        self._radar_data: Optional[np.ndarray] = None
        self._radar3_data: Optional[np.ndarray] = None
        self._radar5_data: Optional[np.ndarray] = None
        self._raw_labels: Optional[np.ndarray] = None
        self._prediction: Optional[np.ndarray] = None

    @property
    def image(self):
        """
        返回图片数据
        :return:image data
        """
        if self._image is not None:
            return self._image
        else:
            self._image = self.get_image()
            return self._image

    @property
    def lidar_data(self):
        """
        返回lidar数据
        :return: Lidar data.
        """
        if self._lidar_data is not None:
            # When the data is already loaded.
            return self._lidar_data
        else:
            # Load data if it is not loaded yet.
            self._lidar_data = self.get_lidar_scan()
            return self._lidar_data

    @property
    def radar_data(self):
        """
        返回 radar数据
        :return: Radar data.
        """
        if self._radar_data is not None:
            # When the data is already loaded.
            return self._radar_data
        else:
            # Load data if it is not loaded yet.
            self._radar_data = self.get_radar_scan()
            return self._radar_data

    @property
    def radar3_data(self):
        """
        返回集成3帧的radar数据
        :return: Radar data.
        """
        if self._radar_data is not None:
            # When the data is already loaded.
            return self._radar_data
        else:
            # Load data if it is not loaded yet.
            self._radar_data = self.get_radar3_scan()
            return self._radar_data

    @property
    def radar5_data(self):
        """
        返回集成5帧的radar数据
        :return: Radar data.
        """
        if self._radar_data is not None:
            # When the data is already loaded.
            return self._radar_data
        else:
            # Load data if it is not loaded yet.
            self._radar_data = self.get_radar5_scan()
            return self._radar_data

    @property
    def raw_labels(self):
        """
    以如下格式返回raw的label:
    类别（Class）：描述物体的类型，如'Car'（汽车）、'Pedestrian'（行人）、'Cyclist'（骑行者）等
    截断（Truncated）：没有被使用，只是为了与KITTI格式兼容
    遮挡（Occluded）：表示遮挡状态的整数值（0、1、2），0表示完全可见，1表示部分遮挡，2表示大部分遮挡
    观测角度（Alpha）：物体的观测角度，范围为[-pi, pi]
    边界框（Bbox）：物体在图像中的二维边界框（从0开始索引），包括左、上、右、下像素坐标
    尺寸（Dimensions）：物体的三维尺寸，包括高度、宽度、长度（以米为单位）
    位置（Location）：物体在相机坐标系中的三维位置坐标（以米为单位）
    旋转（Rotation）：LiDAR传感器围绕-Z轴的旋转角度，范围为[-pi, pi]
        :return: Label data in string format
        """
        if self._raw_labels is not None:
            # When the data is already loaded.
            return self._raw_labels
        else:
            # Load data if it is not loaded yet.
            self._raw_labels = self.get_labels()
            return self._raw_labels

    @property
    def predictions(self):
        """
    以KITTI格式表示的帧的预测信息ncluding:
    类别（Class）：描述物体的类型，如'Car'（汽车）、'Pedestrian'（行人）、'Cyclist'（骑行者）等。
    截断（Truncated）：没有被使用，只是为了与KITTI格式兼容而存在。
    遮挡（Occluded）：表示遮挡状态的整数值（0、1、2），0表示完全可见，1表示部分遮挡，2表示大部分遮挡。
    观测角度（Alpha）：物体的观测角度，范围为[-pi, pi]。
    边界框（Bbox）：物体在图像中的二维边界框（从0开始索引），包括左、上、右、下像素坐标。
    尺寸（Dimensions）：物体的三维尺寸，包括高度、宽度、长度（以米为单位）。
    位置（Location）：物体在相机坐标系中的三维位置坐标（以米为单位）。
    旋转（Rotation）：LiDAR传感器围绕-Z轴的旋转角度，范围为[-pi, pi]。
        :return: Label data in string format
        """
        if self._prediction is not None:
            # When the data is already loaded.
            return self._prediction
        else:
            # Load data if it is not loaded yet.
            self._prediction = self.get_predictions()
            return self._prediction

    def get_image(self) -> Optional[np.ndarray]:
        """
        从数据集中读出image数据
        :return: 以numpy的ndarray格式返回图片内容
        """
        try:
            img = plt.imread(os.path.join(self.data_loactions.camera_dir, f'{self.frame_name}.jpg'))
        except FileNotFoundError:
            logging.error(f"{self.frame_name}.jpg does not exist at location: {self.data_loactions.camera_dir}!")
            return None
        return img

    def set_lidar_scan(self,frame_lidar_data):
        """
        重新设置lidar数据
        :return: Numpy array with lidar data.
        """
        self._lidar_data = frame_lidar_data

    def set_radar_scan(self, frame_radar_data):
        """
        重新设置lidar数据
        :return: Numpy array with lidar data.
        """
        self._radar_data = frame_radar_data


    def get_lidar_scan(self) -> Optional[np.ndarray]:
        """
        从数据集中读出lidar数据
        :return: Numpy array with lidar data.
        """
        try:
            lidar_file = os.path.join(self.data_loactions.lidar_dir, f'{self.file_id}.bin')
            scan = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        except FileNotFoundError:
            logging.error(f"{self.file_id}.bin does not exist at location: {self.data_loactions.lidar_dir}!")
            return None
        return scan

    def get_radar_scan(self) -> Optional[np.ndarray]:
        """
        从数据集中读出radar数据
        :return: Numpy array with radar data.
        """
        try:
            radar_file = os.path.join(self.data_loactions.radar_dir, f'{self.file_id}.bin')
            scan = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)
        except FileNotFoundError:
            logging.error(f"{self.file_id}.bin does not exist at location: {self.data_loactions.radar_dir}!")
            return None
        return scan

    def get_radar3_scan(self) -> Optional[np.ndarray]:
        """
        从数据集中读出3帧的radar数据
        :return: Numpy array with radar data.
        """
        try:
            radar_file = os.path.join(self.data_loactions.radar3_dir, f'{self.file_id}.bin')
            scan = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)
        except FileNotFoundError:
            logging.error(f"{self.file_id}.bin does not exist at location: {self.data_loactions.radar3_dir}!")
            return None
        return scan

    def get_radar5_scan(self) -> Optional[np.ndarray]:
        """
        从数据集中读出5帧的radar数据
        :return: Numpy array with radar data.
        """
        try:
            radar_file = os.path.join(self.data_loactions.radar5_dir, f'{self.file_id}.bin')
            scan = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)
        except FileNotFoundError:
            logging.error(f"{self.file_id}.bin does not exist at location: {self.data_loactions.radar5_dir}!")
            return None
        return scan

    def get_labels(self) -> Optional[List[str]]:
        """
        从txt中读出label信息
        :return: List of strings with label data.
        """
        try:
            label_file = os.path.join(self.data_loactions.label_dir, f'{self.file_id}.txt')
            with open(label_file, 'r') as text:
                labels = text.readlines()
        except FileNotFoundError:
            logging.error(f"{self.file_id}.txt does not exist at location: {self.data_loactions.label_dir}!")
            return None
        return labels

    def get_predictions(self) -> Optional[List[str]]:
        """
         从txt中读出predictions信息
        :return: List of strings with prediction data.
        """
        try:
            label_file = os.path.join(self.data_loactions.pred_dir, f'{self.file_id}.txt')
            with open(label_file, 'r') as text:
                labels = text.readlines()
        except FileNotFoundError:
            logging.error(f"{self.file_id}.txt does not exist at location: {self.data_loactions.pred_dir}!")
            return None
        return labels

if __name__ == '__main__':
    root_path = "../../arun_data/data_set"
    data_locations = FileLocation(root_dir=root_path)
    frame = FrameDataLoader(file_locations=data_locations,frame_name="00000")
    # plt.imshow(frame.image)
    # plt.show()
    # print(frame.lidar_data)
    # print(frame.radar_data)
    # print(frame.radar3_data)
    # print(frame.radar5_data)
    # print(frame.raw_labels)
