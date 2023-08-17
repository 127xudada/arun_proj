from matplotlib import pyplot as plt
from .helpers import plot_boxes, get_2d_label_corners
from .settings import label_color_palette_2d
from arun_lib.data_loader import FrameDataLoader, FrameLabels, FrameTransformMatrix, project_pcl_to_image, min_max_filter


class Visualization2D:
    """
    负责绘制数据集中的一个帧，并可视化其图像及其点云（雷达和/或激光雷达），以及投影和叠加的锚框
    """
    def __init__(self, frame_data_loader: FrameDataLoader,
                 classes_visualized: list = ['Cyclist', 'Pedestrian', 'Car']
                 ):
        """
Constructor of the class, which loads the required frame properties, and creates a copy of the picture data.
        :param frame_data_loader: FrameDataLoader instance.
        :param classes_visualized: A list of classes to be visualized.
        """
        # 创建数据装载器
        self.frame_data_loader = frame_data_loader
        # 创建数据变换矩阵
        self.frame_transformations = FrameTransformMatrix(self.frame_data_loader)
        # ？需要可视化的分类？ 2D只有三个
        self.classes_visualized = classes_visualized
        # 拿出image备份
        self.image_copy = self.frame_data_loader.image

    def plot_gt_labels(self, max_distance_threshold):
        """
        在帧上绘制真实标签（ground truth labels）
        :param max_distance_threshold: 标签绘制的最大距离阈值
        """
        # 根据帧的原始标签数据创建一个FrameLabels对象，获取标签的信息
        frame_labels_class = FrameLabels(self.frame_data_loader.raw_labels)
        # 使用get_2d_label_corners函数根据帧的变换矩阵和标签数据获取每个标签的二维角点坐标
        box_points = get_2d_label_corners(frame_labels_class, self.frame_transformations)
        # 类别过滤。使用filter函数和lambda表达式，将只保留符合self.classes_visualized列表中指定类别的标签角点
        """
            筛选出box_points中label_class在self.classes_visualized列表中的元素，并将它们存储在filtered列表中.
            接受一个元素elem作为输入，并检查elem字典中的label_class键的值是否存在于self.classes_visualized列表中。
            如果存在，则返回True，否则返回False.
            filter函数根据这个匿名函数的返回值对box_points列表进行遍历，并将返回值为True的元素保留下来，构建一个新的列表filtered。
        """
        filtered = list(filter(lambda elem: elem['label_class'] in self.classes_visualized, box_points))
        # 距离过滤。使用filter函数和lambda表达式，将只保留距离小于max_distance_threshold的标签角点
        filtered = list(filter(lambda elem: elem['range'] < max_distance_threshold, filtered))
        # 根据过滤后的标签角点信息，获取对应的颜色和标签列表
        colors = [label_color_palette_2d[v["label_class"]] for v in filtered]
        labels = [d['corners'] for d in filtered]
        # 利用label和color绘制box
        plot_boxes(labels, colors)

    def plot_predictions(self, score_threshold, max_distance_threshold):
        """
        在帧上绘制预测框框（ground truth labels）
        :param score_threshold: The minimum score to be rendered.
        :param max_distance_threshold: The maximum distance where labels are rendered.
        """
        frame_labels_class = FrameLabels(self.frame_data_loader.predictions)
        box_points = get_2d_label_corners(frame_labels_class, self.frame_transformations)
        # Class filter
        filtered = list(filter(lambda elem: elem['label_class'] in self.classes_visualized, box_points))
        # Score filter
        filtered = list(filter(lambda elem: elem['score'] > score_threshold, filtered))
        # Distance filter
        filtered = list(filter(lambda elem: elem['range'] < max_distance_threshold, filtered))
        colors = [label_color_palette_2d[v["label_class"]] for v in filtered]
        labels = [d['corners'] for d in filtered]
        plot_boxes(labels, colors)

    def plot_radar_pcl(self, max_distance_threshold, min_distance_threshold):
        """
        在帧上绘制雷达点云，并根据距离对点进行着色
        :param max_distance_threshold: The maximum distance where points are rendered.
        :param min_distance_threshold: The minimum distance where points are rendered.
        """
        # 使用project_pcl_to_image函数将雷达点云投影到图像上，得到每个点的像素坐标uvs和深度值points_depth。
        uvs, points_depth = project_pcl_to_image(point_cloud=self.frame_data_loader.radar_data,
                                                 t_camera_pcl=self.frame_transformations.t_camera_radar,
                                                 camera_projection_matrix=self.frame_transformations.camera_projection_matrix,
                                                 image_shape=self.frame_data_loader.image.shape)
        # 使用min_max_filter函数对深度值进行过滤，筛选出在指定的最大距离阈值和最小距离阈值之间的点的索引。
        min_max_idx = min_max_filter(points=points_depth,
                                     max_value=max_distance_threshold,
                                     min_value=min_distance_threshold)
        # 保留这些点的像素坐标和深度值
        uvs = uvs[min_max_idx]
        points_depth = points_depth[min_max_idx]
        # 使用plt.scatter函数在图像上绘制散点图
        # 横坐标为uvs中的第一列，纵坐标为uvs中的第二列，颜色根据-points_depth的值来确定，
        # 通过cmap='jet'使用彩虹色映射。
        # 同时，点的大小根据深度值进行缩放，使用公式(70 / points_depth) ** 2计算点的大小
        plt.scatter(uvs[:, 0], uvs[:, 1], c=-points_depth, alpha=0.8, s=(70 / points_depth) ** 2, cmap='jet')

    def plot_lidar_pcl(self,max_distance_threshold, min_distance_threshold):
        """
        在帧上绘制激光雷达点云，并根据距离对点进行着色
        :param max_distance_threshold: The maximum distance where points are rendered.
        :param min_distance_threshold: The minimum distance where points are rendered.
        """
        # 使用project_pcl_to_image函数将激光雷达点云投影到图像上
        # 得到每个点的像素坐标uvs和深度值points_depth
        uvs, points_depth = project_pcl_to_image(point_cloud=self.frame_data_loader.lidar_data,
                                                 t_camera_pcl=self.frame_transformations.t_camera_lidar,
                                                 camera_projection_matrix=self.frame_transformations.camera_projection_matrix,
                                                 image_shape=self.frame_data_loader.image.shape)
        # 使用min_max_filter函数对深度值进行过滤
        # 筛选出在指定的最大距离阈值和最小距离阈值之间的点的索引。
        # 只保留这些点的像素坐标和深度值
        min_max_idx = min_max_filter(points=points_depth,
                                     max_value=max_distance_threshold,
                                     min_value=min_distance_threshold)
        uvs = uvs[min_max_idx]
        points_depth = points_depth[min_max_idx]
        # 使用plt.scatter函数在图像上绘制散点图
        # 横坐标为uvs中的第一列，纵坐标为uvs中的第二列，颜色根据-points_depth的值来确定，
        # 通过cmap='jet'使用彩虹色映射。点的透明度设置为0.4，点的大小为1，表示每个点在图像上都以相同的大小进行绘制
        plt.scatter(uvs[:, 0], uvs[:, 1], c=-points_depth, alpha=0.4, s=1, cmap='jet')

    def draw_plot(self, plot_figure=True,
                  save_figure=True,
                  show_gt: bool = False,
                  show_pred: bool = False,
                  show_lidar: bool = False,
                  show_radar: bool = False,
                  max_distance_threshold: float = 50.0,
                  min_distance_threshold: float = 0.0,
                  score_threshold: float = 0, ):
        """
        绘制带有所需信息的帧
        :param plot_figure: 是否显示图形
        :param save_figure: 是否保存图形
        :param show_gt: 是否绘制地面真实标签
        :param show_pred: 是否绘制预测结果
        :param show_lidar: 是否绘制激光雷达点云
        :param show_radar: 是否绘制雷达点云
        :param max_distance_threshold: 要绘制的物体的最大距离
        :param min_distance_threshold:  要绘制的物体的最小距离
        :param score_threshold:  要绘制的物体的最小得分
        """
        # 创建一个大小为12x8英寸的图形窗口，并设置其分辨率为150dpi
        fig = plt.figure(figsize=(12, 8))
        fig.set_dpi(150)
        # 清除当前的Figure对象中的所有绘图内容
        plt.clf()

        if show_gt:
            self.plot_gt_labels(max_distance_threshold=max_distance_threshold)

        if show_pred:
            self.plot_predictions(max_distance_threshold=max_distance_threshold,
                                  score_threshold=score_threshold)

        if show_lidar:
            self.plot_lidar_pcl(max_distance_threshold=max_distance_threshold,
                                min_distance_threshold=min_distance_threshold)

        if show_radar:
            self.plot_radar_pcl(max_distance_threshold=max_distance_threshold,
                                min_distance_threshold=min_distance_threshold)
        # 绘制图片
        plt.imshow(self.image_copy, alpha=1)
        # 在绘制图形时隐藏坐标轴及其刻度
        plt.axis('off')
        if save_figure:
            plt.savefig(self.frame_data_loader.data_loactions.output_dir + f'{self.frame_data_loader.file_id}.png',
                        bbox_inches='tight', transparent=True)
        if plot_figure:
            plt.show()
        # 关闭图像
        plt.close(fig)
        return


if __name__ == '__main__':
    import os
    from arun_lib.data_loader import FileLocation, FrameDataLoader
    root_path = "../../arun_data/data_set"
    # current_path = os.getcwd()
    output_path = os.path.join(root_path, '..','data_process','2d_image_3radar')
    print(output_path)
    data_locations = FileLocation(root_dir=root_path,output_dir=output_path,radar_type="radar_3frame")
    frame_data = FrameDataLoader(file_locations=data_locations, frame_name="00020")
    vis2d = Visualization2D(frame_data)
    vis2d.draw_plot(show_lidar=True,
                    show_radar=True,
                    show_gt=True,
                    min_distance_threshold=5,
                    max_distance_threshold=20,
                    save_figure=True)