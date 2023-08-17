from typing import Optional, List
import numpy as np
def own_label_genrator(cluster_indices: list, radar_data: np.ndarray):
    """
    生成label以便绘图
    :param cluster_indices: index为每类的类别号，item为对应类的所有radar点
    :param radar_data: 雷达数据源
    :return:
    """
    # 统计所有cluster的总num
    cluster_all_num = len(cluster_indices)
    print("总计类别：{}".format(cluster_all_num))
    # 存放所有类别的x方向最值
    cluster_min_max_x = []
    # 存放所有类别的y方向最值
    cluster_min_max_y = []
    # 存放所有类别的z方向最值
    cluster_min_max_z = []
    cluster_mean_xyz = []
    # 统计单cluster的各自num
    for j, indices in enumerate(cluster_indices):
        cluster_single_num = len(radar_data[indices, :3])
        print("第{}类的rada点数量为{}".format(j, cluster_single_num))
        # x方向最小值
        x_min = np.amin(radar_data[indices, 0])
        # x方向最大值
        x_max = np.amax(radar_data[indices, 0])
        cluster_min_max_x.append((x_min,x_max))
        # y方向最小值
        y_min = np.amin(radar_data[indices, 1])
        # y方向最大值
        y_max = np.amax(radar_data[indices, 1])
        cluster_min_max_y.append((y_min, y_max))
        # z方向最小值
        z_min = np.amin(radar_data[indices, 2])
        # z方向最大值
        z_max = np.amax(radar_data[indices, 2])
        cluster_min_max_z.append((z_min, z_max))
        # 某一类的计算xyz的平均值
        cluster_mean = np.mean(radar_data[indices,:3], axis=0)
        print("本类三个方向的平均值为：{}".format(cluster_mean))
        cluster_mean_xyz.append(cluster_mean)
    # 统计xyz最值
    # print("所有类x方向的最值为：{}".format(cluster_min_max_x))
    cluster_min_max_x = np.array(cluster_min_max_x)
    # print(cluster_min_max_x.shape)
    cluster_min_max_y = np.array(cluster_min_max_y)
    # print(np.array(cluster_min_max_y).shape)
    cluster_min_max_z = np.array(cluster_min_max_z)
    # print(np.array(cluster_min_max_z).shape)
    # 统计每类的长度l，x方向
    cluster_l = cluster_min_max_x[:,1]- cluster_min_max_x[:,0]
    # print(cluster_l)
    # print(cluster_l.shape)
    # 统计每类的宽度w，y方向
    cluster_w = cluster_min_max_y[:, 1] - cluster_min_max_y[:, 0]
    # print(cluster_w)
    # print(cluster_w.shape)
    # 统计每类的高度h，z方向
    cluster_h = cluster_min_max_z[:, 1] - cluster_min_max_z[:, 0]
    # print(cluster_h)
    # print(cluster_h.shape)
    """xyz平均值，即点云类的中心点"""
    cluster_mean_xyz = np.array(cluster_mean_xyz)
    # 抹掉radar的安装高度
    cluster_mean_xyz[:, 1] += 0.3
    print("所有类的点云中心点：\n{}".format(cluster_mean_xyz))
    """lwh值，即点云类的锚框尺寸"""
    # 此处是不是 h
    # cluster_hwl = np.hstack((cluster_h.reshape(-1,1), cluster_w.reshape(-1,1), cluster_l.reshape(-1,1)))
    cluster_hwl = np.hstack((cluster_w.reshape(-1,1), cluster_h.reshape(-1,1),cluster_l.reshape(-1,1)))
    print("所有类的点云锚框尺寸：\n{}".format(cluster_hwl))
    # 将上述xyz拼接成corner的接口
    # 封装自己的labels
    # 创建class列
    label_class = np.arange(cluster_all_num).reshape(-1,1)
    # 创建Truncated列
    label_truncated = np.zeros((cluster_all_num,1),dtype=np.int32)
    # 创建Occluded列
    label_occlude = np.zeros((cluster_all_num,1),dtype=np.int32)
    # 创建Alpha列
    label_alpha = np.zeros((cluster_all_num,1),dtype=np.float32)
    # 创建Bbox框列
    label_bbox = np.zeros((cluster_all_num,4),dtype=np.int32)
    # 创建Dimensons列
    label_dimensons = cluster_hwl
    # 创建Location列
    label_location = cluster_mean_xyz
    # 创建Rotaion列
    label_rotation = np.zeros((cluster_all_num,1),dtype=np.int32)
    # 创建补位列
    label_null = np.ones((cluster_all_num, 1), dtype=np.int32)
    # 封装label
    label = np.hstack((label_class,label_truncated,label_occlude,label_alpha,
                       label_bbox,label_dimensons,label_location,label_rotation,label_null))
    print(label.shape)
    raw_label = [str(row).replace('[', '').replace(']', '') for row in label]
    # print(raw_label)
    return raw_label

class FrameLabels:
    """
    将标签字符串列表转换为Python字典列表
    """
    def __init__(self, raw_labels: List[str]):
        """
        根据包含标签数据的字符串列表创建标签属性
        :param raw_labels: List of strings containing label data.
        """
        # 原始的str类型的labels
        self.raw_labels: List[str] = raw_labels
        # 输出dict类型的labels
        self._labels_dict: Optional[List[dict]] = None

    @property
    def labels_dict(self):
        """
        返回labels_dict
        :return:
        """
        if self._labels_dict is not None:
            # When the data is already loaded.
            return self._labels_dict
        else:
            # Load data if it is not loaded yet.
            self._labels_dict = self.get_labels_dict()
            return self._labels_dict

    def get_labels_dict(self) -> List[dict]:
        """
        返回包含标签数据的字典列表
          *类别（Class）：描述物体的类型，如'Car'（汽车）、'Pedestrian'（行人）、'Cyclist'（骑行者）等。
          *截断（Truncated）：没有被使用，只是为了与KITTI格式兼容而存在。
          *遮挡（Occluded）：表示遮挡状态的整数值（0、1、2），0表示完全可见，1表示部分遮挡，2表示大部分遮挡。
          *观测角度（Alpha）：物体的观测角度，范围为[-pi, pi]。
          *边界框（Bbox）：物体在图像中的二维边界框（从0开始索引），包括左、上、右、下像素坐标。
          *尺寸（Dimensions）：物体的三维尺寸，包括高度、宽度、长度（以米为单位）。h, w, l
          *位置（Location）：物体在相机坐标系中的三维位置坐标（以米为单位）。 x, y, z
          *旋转（Rotation）：LiDAR传感器围绕-Z轴的旋转角度，范围为[-pi, pi]。
        :return: List of dictionaries containing label data.
        """
        # 创建一个空列表labels用于存储结果
        labels = []
        # 从str的list中抽出1列数据
        for act_line in self.raw_labels:  # Go line by line to split the keys
            # 对于每一行，它使用空格分割字符串，将各个字段分配给相应的变量
            act_line = act_line.split()
            print(act_line)
            label, _, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = act_line
            # 高度h、宽度w、长度l、x、y、z坐标以及旋转角度rot和得分score。这些值被转换为浮点数类型。
            h, w, l, x, y, z, rot, score = map(float, [h, w, l, x, y, z, rot, score])
            # 将上述处理过的数据添加到字典中
            labels.append({'label_class': label,
                           'h': h,
                           'w': w,
                           'l': l,
                           'x': x,
                           'y': y,
                           'z': z,
                           'rotation': rot,
                           'score': score}
                          )
        return labels

if __name__ == '__main__':
    from arun_lib.data_loader.file_location import FileLocation
    from arun_lib.data_loader.frame_load import FrameDataLoader
    root_path = "../../arun_data/data_set"
    data_locations = FileLocation(root_dir=root_path)
    frame_data = FrameDataLoader(file_locations=data_locations, frame_name="01047")
    print(frame_data.raw_labels)
    frame_label_dictor = FrameLabels(frame_data.raw_labels)
    print(frame_label_dictor.get_labels_dict())


