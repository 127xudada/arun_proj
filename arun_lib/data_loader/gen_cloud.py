"""
    从指定目录传入bin格式文件
    到指定目录传出pcd格式文件
"""
import os
import pcl
import numpy as np


class Gencloud:
    def __init__(self):
        # 创建cloud对象
        self.cloud = pcl.PointCloud()

    def points2pcd(self,clouds,output_path):
        if not output_path.endswith('.pcd'):
            raise ValueError("Invalid file format. Only .pcd files are outputed.")
        self.cloud.from_array(clouds.astype(np.float32))
        pcl.save(self.cloud,output_path)

    def points2ply(self,clouds,output_path):
        if not output_path.endswith('.ply'):
            raise ValueError("Invalid file format. Only .ply files are outputed.")
        self.cloud.from_array(clouds.astype(np.float32))
        pcl.save(self.cloud,output_path)

    def bin2pcd_lidar(self,input_path,output_path):
        # 异常判断
        if not input_path.endswith('.bin'):
            raise ValueError("Invalid file format. Only .bin files are supported.")
        if not output_path.endswith('.pcd'):
            raise ValueError("Invalid file format. Only .pcd files are outputed.")
        # 读入bin到数组中
        raw_data = np.fromfile(input_path,dtype=np.float32).reshape(-1,4)
        print(raw_data.shape)
        self.cloud.from_array(raw_data[:,:3].astype(np.float32))
        # 转换成点云文件输出
        pcl.save(self.cloud, output_path)

    def bin2ply_lidar(self,input_path,output_path):
        # 异常判断
        if not input_path.endswith('.bin'):
            raise ValueError("Invalid file format. Only .bin files are supported.")
        if not output_path.endswith('.ply'):
            raise ValueError("Invalid file format. Only .ply files are outputed.")
        # 读入bin到数组中
        raw_data = np.fromfile(input_path,dtype=np.float32).reshape(-1,4)
        print(raw_data.shape)
        self.cloud.from_array(raw_data[:,:3].astype(np.float32))
        # 转换成点云文件输出
        pcl.save(self.cloud, output_path)

    def bin2pcd_radar(self,input_path,output_path):
        # 异常判断
        if not input_path.endswith('.bin'):
            raise ValueError("Invalid file format. Only .bin files are supported.")
        if not output_path.endswith('.pcd'):
            raise ValueError("Invalid file format. Only .pcd files are outputed.")
        # 读入bin到数组中
        raw_data = np.fromfile(input_path,dtype=np.float32).reshape(-1,7)
        print(raw_data.shape)
        self.cloud.from_array(raw_data[:,:3].astype(np.float32))
        # 转换成点云文件输出
        pcl.save(self.cloud, output_path)

    def bin2ply_radar(self,input_path,output_path):
        # 异常判断
        if not input_path.endswith('.bin'):
            raise ValueError("Invalid file format. Only .bin files are supported.")
        if not output_path.endswith('.ply'):
            raise ValueError("Invalid file format. Only .ply files are outputed.")
        # 读入bin到数组中
        raw_data = np.fromfile(input_path,dtype=np.float32).reshape(-1,7)
        print(raw_data.shape)
        self.cloud.from_array(raw_data[:,:3].astype(np.float32))
        # 转换成点云文件输出
        pcl.save(self.cloud, output_path)


if __name__ == '__main__':
    # radar测试
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'arun_data', 'data_test', 'ref_cloud')
    bin_cvt = Gencloud()
    name = "00000"
    bin_cvt.bin2pcd_radar(os.path.join(data_dir, name+'.bin'),
                          os.path.join(data_dir, name+'.pcd'))
    bin_cvt.bin2ply_radar(os.path.join(data_dir, name+'.bin'),
                          os.path.join(data_dir, name+'.ply'))
    # # lidar测试
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # data_dir = os.path.join(current_dir, '..', 'arun_data','data_test','ref_cloud')
    # bin_cvt = Gencloud()
    # bin_cvt.bin2pcd_lidar(os.path.join(data_dir, '000017.bin'),
    #                       os.path.join(data_dir, '000017.pcd'))
    # bin_cvt.bin2ply_lidar(os.path.join(data_dir, '000017.bin'),
    #                       os.path.join(data_dir, '000017.ply'))