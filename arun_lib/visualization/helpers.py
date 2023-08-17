import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import k3d
from arun_lib.data_loader.frame_labels import FrameLabels
from arun_lib.data_loader.frame_tf_matrix import FrameTransformMatrix
from arun_lib.data_loader import frame_tf_matrix as transformations

def get_default_camera(pose_transform=np.eye(4, 4)):
    """
    获取在k3d中创建相机视图所需的参数列表
    :param pose_transform: 4x4 transformation matrix of the used coordinate system.
    :return: List required by k3d to create a camera view.
    """
    # Homogenous camera positions
    # 定义了相机的位置、焦点和朝上方向
    camera_pos = [-10, 0, 20, 1]
    camera_focus_point = [10, 0, 1, 1]
    camera_up = [0, 0, 90, 1]
    # 预先定义的默认相机参数
    default_camera = np.array([camera_pos, camera_focus_point, camera_up])
    # 将默认相机参数乘以坐标系变换矩阵，得到相机在给定坐标系下的位置、焦点和朝上方向
    pose_camera = pose_transform.dot(default_camera.T).T
    pose_camera_up = pose_camera[2, :3] - pose_camera[0, :3]
    # 将相机的位置和焦点平铺成一个一维列表，并将朝上方向添加到列表中，返回该列表作为相机视图的参数,以便在k3d中使用这些参数创建相机视图
    return pose_camera[:2, :3].flatten().tolist() + pose_camera_up.tolist()


def get_3d_label_corners(labels: FrameLabels):
    """
    根据给定的FrameLabels对象返回每个标签在帧中的三维角点的列表
    :param labels: FrameLabels object.
          *类别（Class）：描述物体的类型，如'Car'（汽车）、'Pedestrian'（行人）、'Cyclist'（骑行者）等。
          *边界框（Bbox）：物体在图像中的二维边界框（从0开始索引），包括左、上、右、下像素坐标。
          *尺寸（Dimensions）：物体的三维尺寸，包括高度、宽度、长度（以米为单位）。h, w, l
          *位置（Location）：物体在相机坐标系中的三维位置坐标（以米为单位）。 x, y, z
    :return: List of 3d corners.
    """
    # 空的label_corners列表，用于存储每个标签的三维角点信息
    label_corners = []

    for label in labels.labels_dict:
        # 函数根据标签的长度(l)、宽度(w)和高度(h)计算出8个角点的x、y、z坐标值，并将它们分别存储在x_corners、y_corners和z_corners列表中
        """
            1-------2
            -
            -
            -
        
        """

        x_corners = [label['l'] / 2,
                     label['l'] / 2,
                     -label['l'] / 2,
                     -label['l'] / 2,
                     label['l'] / 2,
                     label['l'] / 2,
                     -label['l'] / 2,
                     -label['l'] / 2]
        y_corners = [label['w'] / 2,
                     -label['w'] / 2,
                     -label['w'] / 2,
                     label['w'] / 2,
                     label['w'] / 2,
                     -label['w'] / 2,
                     -label['w'] / 2,
                     label['w'] / 2]
        z_corners = [0,
                     0,
                     0,
                     0,
                     label['h'],
                     label['h'],
                     label['h'],
                     label['h']]

        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        # 定义一个字典列表，将每个标签的类别、置信度和corners_3d添加到label_corners列表中作为一个字典项。
        label_corners.append({'label_class': label['label_class'],
                              'score': label['score'],
                              'corners_3d': corners_3d})
    # 返回label_corners列表，其中包含每个标签的类别、置信度和三维角点的信息
    return label_corners


def get_transformed_3d_label_corners(labels: FrameLabels, transformation, t_camera_lidar):
    """
    将3D点转换到camera坐标系下，用于将给定标签的3D角点坐标根据特定的变换进行转换
    :param labels:
    :param transformation:
    :param t_camera_lidar:
    :return:
    """
    # 解析出来了立体8点
    corners_3d = get_3d_label_corners(labels)
    # 坐标变换后的立体8点
    transformed_3d_label_corners = []
    # 遍历字典
    for index, label in enumerate(labels.labels_dict):
        # 计算旋转角度，将标签的旋转角度恢复为原始角度。这是通过将标签的旋转角度加上π/2并取负来实现的
        rotation = -(label['rotation'] + np.pi / 2)  # undo changes made to rotation
        # 创建一个旋转矩阵rot_matrix，用于将标签的3D角点坐标旋转回原始角度
        rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                               [np.sin(rotation), np.cos(rotation), 0],
                               [0, 0, 1]])

        # 根据相机到激光雷达的变换矩阵t_camera_lidar将标签的中心点从相机坐标系转换到激光雷达坐标系。
        # 通过将标签中心点的坐标向量乘以t_camera_lidar的逆矩阵实现的。
        center = (np.linalg.inv(t_camera_lidar) @ np.array([label['x'],
                                                             label['y'],
                                                             label['z'],
                                                             1]))[:3]

        # 根据旋转矩阵和中心点，对每个标签的原始角点坐标执行旋转和平移变换
        new_corner_3d = np.dot(rot_matrix, corners_3d[index]['corners_3d']).T + center
        # 将转换后的3D角点坐标与一个全为1的列向量进行连接，得到齐次坐标形式的转换后的角点坐标
        new_corners_3d_hom = np.concatenate((new_corner_3d, np.ones((8, 1))), axis=1)
        # 使用给定的变换矩阵transformation对转换后的角点坐标执行齐次变换
        new_corners_3d_hom = transformations.homogeneous_transformation(new_corners_3d_hom, transformation)
        # print("变换后的矩阵如下\n{}".format(new_corners_3d_hom))
        # 函数返回包含所有标签的转换后的3D角点数据的列表
        transformed_3d_label_corners.append({'label_class': label['label_class'],
                                             'corners_3d_transformed': new_corners_3d_hom,
                                           'score': label['score']})
    # 函数返回包含所有标签的转换后的3D角点数据的列表
    return transformed_3d_label_corners


def get_2d_label_corners(labels: FrameLabels, transformations_matrix: FrameTransformMatrix):
    """
    计算给定标签的2D角点在图像中的坐标
    :param labels:
    :param transformations_matrix:
    :return: 输出是一列表，其中每个元素是一个字典，包含了每个标签的类别、2D角点在图像中的坐标、置信度和距离
    """
    bboxes = []
    # 函数调用get_3d_label_corners函数获取原始标签的3D角点坐标，将其存储在corners_3d变量中。
    corners_3d = get_3d_label_corners(labels)
    # 通过遍历labels.labels_dict中的每个标签来处理每个标签的数据
    for index, label in enumerate(labels.labels_dict):
        # 计算旋转角度，将标签的旋转角度恢复为原始角度
        rotation = -(label['rotation'] + np.pi / 2)  # undo changes made to rotation
        # 创建一个旋转矩阵rot_matrix，用于将标签的3D角点坐标旋转回原始角度
        rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                               [np.sin(rotation), np.cos(rotation), 0],
                               [0, 0, 1]])
        # 根据激光雷达到相机的变换矩阵transformations_matrix.t_lidar_camera将标签的中心点从激光雷达坐标系转换到相机坐标系。
        # 这是通过将标签中心点的坐标向量乘以transformations_matrix.t_lidar_camera实现的。
        center = (transformations_matrix.t_lidar_camera @ np.array([label['x'],
                                                                    label['y'],
                                                                    label['z'],
                                                                    1]))[:3]
        # 根据旋转矩阵和中心点，对每个标签的原始角点坐标执行旋转和平移变换。这是通过将旋转矩阵与原始角点坐标进行矩阵乘法，并将结果加上中心点坐标实现的
        new_corner_3d = np.dot(rot_matrix, corners_3d[index]['corners_3d']).T + center
        # 将转换后的3D角点坐标与一个全为1的列向量进行连接，得到齐次坐标形式的转换后的角点坐
        new_corners_3d_hom = np.concatenate((new_corner_3d, np.ones((8, 1))), axis=1)
        # 调用homogeneous_transformation函数，使用给定的变换矩阵transformations_matrix.t_camera_lidar
        # 对转换后的角点坐标执行齐次变换，将其转换到相机坐标系
        new_corners_3d_hom = transformations.homogeneous_transformation(new_corners_3d_hom, transformations_matrix.t_camera_lidar)
        # 根据相机的投影矩阵transformations_matrix.camera_projection_matrix将转换后的角点坐标投影到图像平面，得到角点在图像中的坐标
        corners_img = np.dot(new_corners_3d_hom, transformations_matrix.camera_projection_matrix.T)
        # 对角点坐标进行归一化，将其除以其齐次坐标的第三个元素，以得到2D图像坐标
        corners_img = (corners_img[:, :2].T / corners_img[:, 2]).T
        # 将角点坐标转换为列表形式
        corners_img = corners_img.tolist()
        # 并计算标签中心点到相机的距离
        distance = np.linalg.norm((label['x'], label['y'], label['z']))
        # 将标签的类别、角点坐标、置信度和距离整合成一个字典，并将该字典添加到bboxes列表中。
        bboxes.append({'label_class': label['label_class'],
                       'corners': corners_img,
                       'score': label['score'],
                       'range': distance})
    # 按照距离进行排序，并返回排序后的结果
    bboxes = sorted(bboxes, key=lambda d: d['range'])
    return bboxes


def mask_pcl(scan, scan_C2, inds, scan_c2_depth):
    """
    根据给定的条件对点云数据进行筛选.
    函数根据点云中点的三维坐标的欧氏距离是否小于30来创建一个布尔掩码。
    如果点的欧氏距离小于30，则对应的布尔值为True，否则为False
    :param scan:
    :param scan_C2:
    :param inds:
    :param scan_c2_depth:
    :return:
    """
    # 根据点云中点的三维坐标的欧氏距离是否小于30来创建一个布尔掩码。  只看30m??
    # 如果点的欧氏距离小于30，则对应的布尔值为True，否则为False
    mask = np.linalg.norm(scan[:, :3], axis=1) < 30
    # 使用这个布尔掩码来筛选原始的点云scan、scan_C2、inds和scan_c2_depth。
    # 只有在对应的掩码值为True的位置，才会被保留下来
    return scan[mask], scan_C2[mask], inds[mask], scan_c2_depth[mask]


def line(p1, p2, color):
    """
    使用 plt.gca().add_line() 函数向当前的坐标轴添加一条线段。(get current axes)
    Line2D() 函数用于创建一条线段对象，    通过传递起始点和终点的坐标、颜色和线宽等参数来定义线段的属性。
    然后，使用 add_line() 方法将线段对象添加到当前的坐标轴中
    :param p1:
    :param p2:
    :param color:
    :return:
    """
    plt.gca().add_line(
        Line2D((p1[0], p2[0]), (p1[1], p2[1]), color=color, linewidth=1))


def face(corners: np.ndarray, color: tuple, alpha: float = 0.3):
    """
    定义了一个函数 face，用于绘制一个面（多边形）
    :param corners: 包含多个顶点坐标的 NumPy 数组，形状为 (n, 2)，其中 n 是顶点的数量。每个顶点都由 (x, y) 坐标表示
    :param color: 一个表示面颜色的元组，可以是 RGB 形式的颜色值或者字符串表示的颜色名称
    :param alpha: alpha：可选参数，表示面的透明度，默认为 0.3。
    :return:
    """
    # 取出corners中的x,y坐标
    xs = corners[:, 0]
    ys = corners[:, 1]
    # 使用 plt.fill() 函数绘制面，将 xs 和 ys 作为顶点的 x 坐标和 y 坐标，color 表示面的颜色，alpha 表示面的透明度
    plt.fill(xs, ys, color=color, alpha=alpha)


def plot_boxes(boxes: list, colors=None):
    """
    plot_boxes，用于绘制一组框（box）
    :param boxes:   一个包含多个框的列表，每个框由一组顶点坐标组成,以特定的顺序存储在 corners_img 数组中
    :param colors:
    :return: 可选参数，表示每个框的颜色。如果未提供颜色列表，则默认使用白色
    """
    # 遍历 boxes 列表中的每个框。
    for j in range(len(boxes)):
        # 提取当前框的顶点坐标，存储在 corners_img 数组中
        corners_img = np.array(boxes[j])
        if colors is not None:
            color = colors[j]
        else:
            # 如果没颜色，则设置为白色
            color = (1.0, 1.0, 1.0)
        # 如果是白色
        if color == (1.0, 1.0, 1.0):
            # 透明度0.15
            alpha = 0.15
        else:
            alpha = 0.2
        # 根据框的顶点坐标，使用 face 函数绘制框的六个面： 四个侧面和两个底面
        # 前四个面使用 corners_img[:4] 和颜色 color 进行绘制
        # 后四个面使用 corners_img[4:] 和颜色 color 进行绘制
        face(corners_img[:4], color, alpha)
        face(corners_img[4:], color, alpha)
        # 其余两个面使用特定的顶点坐标组合调用 face 函数进行绘制
        face(np.array([corners_img[0], corners_img[1], corners_img[5], corners_img[4]]), color, alpha)
        face(np.array([corners_img[1], corners_img[2], corners_img[6], corners_img[5]]), color, alpha)
        face(np.array([corners_img[2], corners_img[3], corners_img[7], corners_img[6]]), color, alpha)
        face(np.array([corners_img[0], corners_img[3], corners_img[7], corners_img[4]]), color, alpha)
    return


def k3d_get_axes(hom_transform_matrix=np.eye(4, dtype=float), axis_length=1.0):
    """
    生成基于给定变换矩阵的坐标轴系统
    :param hom_transform_matrix: 一个4x4的齐次变换矩阵，用于表示坐标系的变换.
    :param axis_length: 坐标轴的长度，默认为1.0
    :return: k3d vector of the axis system.
    """
    # 创建一个3x4的齐次向量 hom_vector，其中每行表示一个坐标轴的单位向量，
    # 并将其与一个齐次坐标 [axis_length, 0, 0, 1]、[0, axis_length, 0, 1] 和 [0, 0, axis_length, 1] 连接起来
    hom_vector = np.asarray([[axis_length, 0, 0, 1],
                         [0, axis_length, 0, 1],
                         [0, 0, axis_length, 1]],
                        dtype=float)
    # 根据变换矩阵 hom_transform_matrix，计算出三个坐标轴的起点和终点坐标
    origin = [hom_transform_matrix[:3, 3]] * 3
    # 使用起点和终点坐标以及预定义的颜色值，创建一个 k3d 的矢量对象 pose_axes，表示坐标轴系统。
    axis = hom_transform_matrix.dot(hom_vector.T).T[:, :3]
    pose_axes = k3d.vectors(
        origins=origin,
        vectors=axis[:, :3] - origin,
        colors=[0xFF0000, 0xFF0000, 0x00FF00, 0x00FF00, 0x0000FF, 0x0000FF])
    # 返回生成的 pose_axes
    return pose_axes


def k3d_plot_box(plot, box_corners, color, width):
    """
    用于在 k3d 中绘制一个立方体框
    :param plot:k3d 绘图对象，用于绘制立方体框
    :param box_corners:一个包含立方体框八个角点坐标的数组
    :param color:立方体框的颜色
    :param width:
    :return:立方体框的线宽
    """
    lines = [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [3, 7], [2, 6]]
    # 迭代处理 lines 列表中的线段索引，将对应的角点坐标传递给 k3d.line 函数，以绘制立方体框的线段
    for plot_line in lines:
        plot += k3d.line(box_corners[plot_line, 0:3], color=color, width=width)

def k3d_own_plot_box(plot,box_corners,color):
    pass


def get_radar_velocity_vectors(pc_radar, compensated_radial_velocity):
    """
    函数用于计算雷达速度向量
    :param pc_radar:雷达点云的坐标
    :param compensated_radial_velocity:经过校正的径向速度
    :return:
    """
    # 首先计算雷达点云的径向单位向量 radial_unit_vectors，即将每个点的坐标除以其模长
    radial_unit_vectors = pc_radar / np.linalg.norm(pc_radar, axis=1, keepdims=True)
    # 通过将校正的径向速度乘以径向单位向量，得到速度向量 velocity_vectors
    velocity_vectors = compensated_radial_velocity[:, None] * radial_unit_vectors
    return velocity_vectors

if __name__ == '__main__':
    pass