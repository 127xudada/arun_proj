# 给label分配颜色
"""
Cyclist:红
Pedestrian:绿
Car:蓝
"""
label_color_palette_2d = {'Cyclist': (1.0, 0.0, 0.0),
                       'Pedestrian': (0.0, 1.0, 0.0),
                       'Car': (0.0, 0.0, 1.0)}
# 轴长度
axis_length_3d = 1
axis_label_size = 0.5
# 设置颜色
label_color_palette_3d = {
            "Car": 0xFF0000,
            "Pedestrian": 0x00FF00,
            "Cyclist": 0x0000FF,
            "bicycle": 0x0000FF,
            "DontCare": 0xAAAAAA,
            "moped_scooter": 0xAAAAAA,
            "rider": 0xFF0000,
            "bicycle_rack": 0xAAAAAA,
            "ride_other": 0xAAAAAA,
            "motor": 0xFF9000,
            "truck": 0xFF0090
}
# 设置线宽
label_line_width_3d = {
            "Car": 0.05,
            "Pedestrian": 0.05,
            "Cyclist": 0.02,
            "bicycle": 0.05,
            "DontCare": 0.01,
            "moped_scooter": 0.05,
            "rider": 0.02,
            "bicycle_rack": 0.01,
            "ride_other": 0.01,
            "motor": 0.01,
            "truck": 0.01
        }
# radar的颜色：红色
radar_plot_color_3d = 0xFF6347
# lidar的颜色：蓝色
lidar_plot_color_3d = 0x0000FF
# radar的相对速度的颜色
radar_velocity_color_3d = 0x9400D3
# radar和lidar的点云大小
radar_pcl_size = 0.3
lidar_pcl_size = 0.08
