from .helpers import *
from arun_lib.data_loader import FrameDataLoader, FrameTransformMatrix, FrameLabels, transform_pcl,own_label_genrator
from .settings import *
import k3d
import numpy as np

class Visualization3D:
    """
    使用k3d库绘制动态交互的3D图
    """
    def __init__(self, frame_data: FrameDataLoader, origin='camera'):
        """
        构造函数，用于在所请求的坐标系中准备3D绘图。默认以相机坐标系为中心
        :param frame_data:
        """
        # 存储绘图对象
        self.plot = None
        # 存储FrameDataLoader实例的变量，用于获取帧数据
        self.frame_data = frame_data
        # 获取坐标系之间的变换矩阵
        self.frame_transforms = FrameTransformMatrix(self.frame_data)
        # 存储传入的origin参数的值，表示所请求的坐标系原点
        self.origin = origin
        # 使用相机坐标系作为基准
        if self.origin == 'camera':
            # 将变换矩阵设为单位矩阵（np.eye(4, dtype=float)），表示相机坐标系与自身之间的转换
            self.transform_matrices = {
                'camera': np.eye(4, dtype=float),
                'lidar': self.frame_transforms.t_camera_lidar,
                'radar': self.frame_transforms.t_camera_radar
            }
        # 使用激光雷达坐标系作为基准
        elif self.origin == 'lidar':
            # 将变换矩阵设置为激光雷达坐标系与相机坐标系之间的变换矩阵
            self.transform_matrices = {
                'camera': self.frame_transforms.t_lidar_camera,
                'lidar': np.eye(4, dtype=float),
                'radar': self.frame_transforms.t_lidar_radar
            }
        # 使用雷达坐标系作为基准
        elif self.origin == 'radar':
            # 将变换矩阵设置为雷达坐标系与相机坐标系之间的变换矩阵
            self.transform_matrices = {
                'camera': self.frame_transforms.t_radar_camera,
                'lidar': self.frame_transforms.t_radar_lidar,
                'radar': np.eye(4, dtype=float)
            }
        else:
            raise ValueError("Origin must be camera, lidar or radar!")

    def __call__(self, radar_origin_plot: bool = False,
                 lidar_origin_plot: bool = False,
                 camera_origin_plot: bool = False,
                 lidar_points_plot: bool = False,
                 radar_points_plot: bool = False,
                 radar_velocity_plot: bool = False,
                 plot_annotations: bool = False):
        # 定义了类的__call__方法，该方法允许将类的实例像函数一样进行调用。它接受多个布尔类型的参数，用于指定要在3D图中绘制的内容
        self.draw_plot(radar_origin_plot,
                       lidar_origin_plot,
                       camera_origin_plot,
                       lidar_points_plot,
                       radar_points_plot,
                       radar_velocity_plot,
                       plot_annotations)

    def plot_radar_origin(self,
                          label: bool = True,
                          color: int = radar_plot_color_3d,
                          axis_length: float = axis_length_3d,
                          label_size: float = axis_label_size):
        """
        在所请求的坐标系中绘制雷达原点
        :param axis_length: 轴的长度
        :param label: 布尔值，确定是否显示标签
        :param color: 标签的颜色，以整数表示
        :param label_size: 标签的大小
        """
        # 调用k3d_get_axes函数绘制坐标轴，以雷达原点为中心，并根据提供的axis_length确定轴的长度
        self.plot += k3d_get_axes(self.transform_matrices['radar'], axis_length)
        # 如果label参数为True，则使用k3d.text函数在雷达原点处绘制一个文本标签
        # 文本内容为"radar"，颜色为color，大小为label_size。
        if label:
            self.plot += k3d.text("radar",
                                  position=self.transform_matrices['radar'][0:3, 3],
                                  color=color,
                                  size=label_size)

    def plot_lidar_origin(self,
                          label: bool = True,
                          color: int = lidar_plot_color_3d,
                          axis_length: float = axis_length_3d,
                          label_size: float = axis_label_size):
        """
        同上
        :param axis_length: Vector length of the axis.
        :param label: Bool which sets if the label should be displayed.
        :param color: Color of the label in int.
        :param label_size: Size of the label.
        """
        self.plot += k3d_get_axes(self.transform_matrices['lidar'], axis_length)
        if label:
            self.plot += k3d.text("lidar",
                                  position=self.transform_matrices['lidar'][0:3, 3],
                                  color=color,
                                  size=label_size)

    def plot_camera_origin(self,
                           label: bool = True,
                           color: int = lidar_plot_color_3d,
                           axis_length: float = axis_length_3d,
                           label_size: float = axis_label_size):
        """
        同上
        :param axis_length: Vector length of the axis.
        :param label: Bool which sets if the label should be displayed.
        :param color: Color of the label in int.
        :param label_size: Size of the label.
        """
        self.plot += k3d_get_axes(self.transform_matrices['camera'], axis_length)
        if label:
            self.plot += k3d.text("camera",
                                  position=self.transform_matrices['camera'][0:3, 3],
                                  color=color,
                                  size=label_size)

    def plot_lidar_points(self,
                          pcl_size: float = lidar_pcl_size,
                          color: int = lidar_plot_color_3d):
        """
        在请求的帧中绘制激光雷达点云
        :param pcl_size: 图中点云粒子的大小.
        :param color: 图中点云粒子的颜色.
        """
        # 调用transform_pcl函数将激光雷达数据点云转换到相机坐标系中
        lidar_points_camera_frame = transform_pcl(points=self.frame_data.lidar_data,
                                                  transform_matrix=self.transform_matrices['lidar'])
        # 使用k3d.points函数将转换后的点云在3D图中以位置
        # lidar_points_camera_frame、点的大小为pcl_size、颜色为color的形式绘制出来。
        self.plot += k3d.points(positions=np.asarray(lidar_points_camera_frame[:, :3], dtype=float),
                                point_size=pcl_size,
                                color=color)

    def plot_radar_points(self,
                          pcl_size: float = radar_pcl_size,
                          color: int = radar_plot_color_3d
                          ):
        """
        在请求的帧中绘制雷达点云
        :param pcl_size: Size of the pcl particles in the graph.
        :param color: Color of the pcl particles in the graph.
        """
        radar_points_camera_frame = transform_pcl(points=self.frame_data.radar_data,
                                                  transform_matrix=self.transform_matrices['radar'])
        # 使用k3d.points函数将转换后的点云在3D图中绘制
        self.plot += k3d.points(positions=np.asarray(radar_points_camera_frame[:, :3], dtype=float),
                                point_size=pcl_size,
                                color=color)

    def plot_color_radar_points(self,cluster_indices: list,
                                pcl_size: float = radar_pcl_size):
        """
        将radar点，根据不同颜色绘制出来
        :param cluster_indices: list对象，index代表第几类，item表示此类的像素点
        :param pcl_size: 绘制radar点的大小
        :return:
        """
        # 先将所有radar的点转到pcl坐标系下
        radar_points_camera_frame = transform_pcl(points=self.frame_data.radar_data,
                                                  transform_matrix=self.transform_matrices['radar'])
        # 从radar_points_camera_frame取出所有坐标点数据
        radar_points = radar_points_camera_frame[:, :3]
        # 将radar_points从NX3拓展成NX4存放一列颜色值，初始化颜色值为1
        color_radar_points = np.hstack((radar_points, np.ones((radar_points.shape[0], 1), dtype=np.float32)))
        # 初始化颜色为1000
        color_radar_points[:,3] = 0x9400D3
        # colors = np.ones((radar_points.shape[0], 1))
        # 从cluster_indices中取出类和对应的点

        print('cluster_num : ' + str(len(cluster_indices)) + "  .")
        # 将每个点对应的颜色值放到第4列中
        for j, indices in enumerate(cluster_indices):
            color_radar_points[indices, 3] = (j * 2) * 10000  # 此处计算颜色值可以用别的算法
        # colors=color_radar_points[:,3]
        # print(colors.shape)
        # 使用k3d.points函数将转换后的点云在3D图中绘制
        self.plot += k3d.points(positions=np.asarray(radar_points_camera_frame[:, :3], dtype=float),
                                point_size=pcl_size,
                                colors=color_radar_points[:,3])


    def plot_radar_radial_velocity(self, color: int = radar_velocity_color_3d):
        """
        在请求的帧中绘制雷达点云的径向速度矢量
        :param color: Color of the vector.
        """
        # 从雷达数据集中取出第5列的相对速度
        compensated_radial_velocity = self.frame_data.radar_data[:, 5]
        # 将雷达数据点云转换到相机坐标系中
        radar_points_camera_frame = transform_pcl(points=self.frame_data.radar_data,
                                                  transform_matrix=self.transform_matrices['radar'])
        # 从雷达数据集中取出点云数据
        pc_radar = radar_points_camera_frame[:, 0:3]
        # 基于雷达点云和补偿径向速度计算雷达径向速度矢量。速度矢量存储在velocity_vectors变量中
        velocity_vectors = get_radar_velocity_vectors(pc_radar, compensated_radial_velocity)
        # 将速度矢量绘制在3D图
        self.plot += k3d.vectors(origins=pc_radar, vectors=velocity_vectors, color=color)

    def plot_own_annotations(self,cluster_indices: list,class_colors=0x0000F0,class_width=0.08):
        """
        用统一颜色绘制3D框框
        :param cluster_indices:
        :param class_colors:
        :param class_width:
        :return:
        """
        # 从own_labels中获取信息
        radar_points_camera_frame = transform_pcl(points=self.frame_data.radar_data,
                                                  transform_matrix=self.transform_matrices['radar'])
        raw_label = own_label_genrator(cluster_indices,radar_points_camera_frame)
        labels: FrameLabels = FrameLabels(raw_label)
        # 把基于标签、Lidar的变换矩阵和相机与Lidar的变换矩阵获取转换后的3D标签框放入bboxes。
        # bboxes = get_transformed_3d_label_corners(labels, self.transform_matrices['lidar'],
        #                                           self.frame_transforms.t_camera_lidar)
        bboxes = get_transformed_3d_label_corners(labels, self.transform_matrices['radar'],
                                                  self.frame_transforms.t_camera_radar)
        # 循环遍历每个标签框，获取对象类别、对象类别的颜色和线宽
        for box in bboxes:
            object_class = box['label_class']
            object_class_color = (int(float(object_class))*2)*10000
            # object_class_color =int(object_class)*class_colors+1000
            object_class_width = class_width
            corners_object = box['corners_3d_transformed']
            # 使用上述参数，在3D图中绘制标签框
            k3d_plot_box(self.plot, corners_object, object_class_color, object_class_width)



    def plot_annotations(self, class_colors=label_color_palette_3d, class_width=label_line_width_3d):
        """
        在请求的帧中绘制锚框
        :param class_colors: 锚框颜色的字典.
        :param class_width: 锚框线宽的字典.
        """
        # 从帧数据中获取标签信息
        labels: FrameLabels = FrameLabels(self.frame_data.raw_labels)
        # 把基于标签、Lidar的变换矩阵和相机与Lidar的变换矩阵获取转换后的3D标签框放入bboxes。
        bboxes = get_transformed_3d_label_corners(labels,self.transform_matrices['lidar'],
                                                  self.frame_transforms.t_camera_lidar)
        # 循环遍历每个标签框，获取对象类别、对象类别的颜色和线宽
        for box in bboxes:
            object_class = box['label_class']
            object_class_color = class_colors[object_class]
            object_class_width = class_width[object_class]
            corners_object = box['corners_3d_transformed']
            # 使用上述参数，在3D图中绘制标签框
            k3d_plot_box(self.plot, corners_object, object_class_color, object_class_width)

    def draw_plot(self,
                  radar_origin_plot: bool = False,
                  lidar_origin_plot: bool = False,
                  camera_origin_plot: bool = False,
                  lidar_points_plot: bool = False,
                  radar_points_plot: bool = False,
                  radar_color_points_plot: bool = False,
                  cluster_indices: list = None,
                  radar_velocity_plot: bool = False,
                  own_annotations_plot: bool = False,
                  annotations_plot: bool = False,
                  write_to_html: bool = False,
                  html_name: str = "example",
                  grid_visible: bool = False,
                  auto_frame: bool = False,
                  ):
        """
        显示带有指定参数的图
        :param auto_frame: 当设置为True时，自动调整图形尺寸
        :param grid_visible: 是否显示图形的网格背景
        :param radar_origin_plot: 是否绘制雷达原点坐标轴
        :param lidar_origin_plot: 是否绘制激光雷达原点坐标轴
        :param camera_origin_plot: 是否绘制相机原点坐标轴
        :param lidar_points_plot: 是否绘制激光雷达点云
        :param radar_points_plot: 是否绘制雷达点云
        :param radar_color_points_plot: 是否绘制不同类别的彩色点云
        :param cluster_indices: 待绘制的点云数据
        :param radar_velocity_plot: 是否绘制雷达径向速度向量
        :param annotations_plot: 是否绘制锚框
        :param write_to_html: 是否将图形写入HTML文件
        :param html_name: 如果写入到磁盘上的HTML文件的名称
        """
        # 调用k3d的画图
        self.plot = k3d.plot(camera_auto_fit=auto_frame, axes_helper=0.0, grid_visible=grid_visible)
        # 绘制camera、lidar、radar的原点
        if radar_origin_plot:
            self.plot_radar_origin()
        if lidar_origin_plot:
            self.plot_lidar_origin()
        if camera_origin_plot:
            self.plot_camera_origin()
        # 绘制radar和lidar的点云
        if lidar_points_plot:
            self.plot_lidar_points()
        if radar_points_plot:
            self.plot_radar_points()
        # 绘制radar彩色点云
        if radar_color_points_plot:
            self.plot_color_radar_points(cluster_indices)
        # 绘制radar的镜像速度矢量
        if radar_velocity_plot:
            self.plot_radar_radial_velocity()
        # 绘制锚框
        if annotations_plot:
            self.plot_annotations()
        if own_annotations_plot:
            self.plot_own_annotations(cluster_indices=cluster_indices)
        # 是否自动调整图形尺寸
        if not auto_frame:
            self.plot.camera = get_default_camera(self.transform_matrices['lidar'])
        # 将k3d显示出来
        self.plot.display()
        # 是否写入html中
        if write_to_html:
            self.plot.snapshot_type = 'inline'
            data = self.plot.get_snapshot()
            with open(f'{html_name}_'+self.frame_data.file_id+'.html', 'w') as f:
                f.write(data)

if __name__ == '__main__':
    from arun_lib.data_loader import FileLocation, FrameDataLoader
    root_path = "../../arun_data/data_set"
    output_path = "3d_image_5frames"
    data_locations = FileLocation(root_dir=root_path, output_dir=output_path,radar_type='radar_5frame')
    frame_data = FrameDataLoader(file_locations=data_locations, frame_name="00020")
    vis3d = Visualization3D(frame_data)
    vis3d.draw_plot(radar_origin_plot=True,
                    lidar_origin_plot=True,
                    camera_origin_plot=True,
                    lidar_points_plot=True,
                    radar_points_plot=True,
                    radar_velocity_plot=True,
                    annotations_plot=True,
                    grid_visible=True,
                    write_to_html=True,
                    html_name= output_path )
