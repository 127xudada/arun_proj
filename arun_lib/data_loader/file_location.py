import os

class FileLocation:
    """
    找到数据集的具体位置
    """
    def __init__(self, root_dir: str, output_dir: str = None, radar_type:str = "radar_1frame", frame_set_path: str = None, pred_dir: str = None):
        """
        创建Filelocation 并且初始化
        :param root_dir: 数据集的根目录名
        :param output_dir: 可选参数，生成输出（如图片）的位置。
        :param frame_set_path: 可选参数，生成输出的文本文件的位置。
        :param pred_dir: 可选参数，预测标签的位置
        """
        # 初始化变量
        self.root_dir: str = root_dir
        self.output_dir: str = output_dir
        self.frame_set_path: str = frame_set_path
        self.pred_dir: str = pred_dir
        # 自动定义的变量。可以在这里自定义子文件夹的位置。
        # 当前的定义基于推荐的位置。
        # 装载照片的目录
        self.camera_dir = os.path.join(self.root_dir, 'lidar', 'training', 'image_2')
        # 装载lidar数据的目录
        self.lidar_dir = os.path.join(self.root_dir, 'lidar', 'training', 'velodyne')
        # 装载lidar标定数据的目录
        self.lidar_calib_dir = os.path.join(self.root_dir, 'lidar', 'training', 'calib')
        # 装载单帧radar数据的目录
        if radar_type == "radar_1frame":
            self.radar_dir = os.path.join(self.root_dir, 'radar', 'training', 'velodyne')
        if radar_type == "radar_3frame":
            self.radar_dir = os.path.join(self.root_dir, 'radar_3frames', 'training', 'velodyne')
        if radar_type == "radar_5frame":
            self.radar_dir = os.path.join(self.root_dir, 'radar_5frames', 'training', 'velodyne')

        self.radar3_dir = os.path.join(self.root_dir, 'radar_3frames', 'training', 'velodyne')
        self.radar5_dir = os.path.join(self.root_dir, 'radar_5frames', 'training', 'velodyne')
        self.radar_calib_dir = os.path.join(self.root_dir, 'radar', 'training', 'calib')
        # 装载位姿数据的目录
        self.pose_dir = os.path.join(self.root_dir, 'lidar', 'training', 'pose')
        self.pose_calib_dir = os.path.join(self.root_dir, 'lidar', 'training', 'calib')
        # 装载标签数据的目录？？ 标签的格式含义是什么？？人为画上去的点？？
        self.label_dir = os.path.join(self.root_dir, 'lidar', 'training', 'label_2')

if __name__ == '__main__':
    root_path = "../../arun_data/data_set"
    data_locations = FileLocation(root_dir=root_path)
    print(f"Lidar directory: {data_locations.lidar_dir}")
    print(f"Radar directory: {data_locations.radar_dir}")
    print(f"Radar 3frames directory: {data_locations.radar3_dir}")
    print(f"Radar 5frames directory: {data_locations.radar5_dir}")