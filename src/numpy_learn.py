import torch
import numpy as np

def print_mac_info():
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())

# 声明一个图片数组
def clare_pic_numpy():
    # 创建一个1x3x256x256的数组，其中所有元素初始化为0
    array = np.zeros((1, 3, 256, 256), dtype=np.uint8)

    # 如果需要，你可以将数组中的元素赋予不同的值
    # 例如，将第一个通道的第一个像素点设置为红色（255, 0, 0）
    array[0, 0, 0, 0] = 255  # 红色通道
    array[0, 1, 0, 0] = 0  # 绿色通道
    array[0, 2, 0, 0] = 0  # 蓝色通道
    print(array.shape)
    print(array.ndim)

def clare_pic_tensor():
    # 创建一个1x3x256x256的张量，其中所有元素初始化为0
    tensor = torch.zeros(1, 3, 256, 256)

    # 如果需要，你可以将张量中的元素赋予不同的值
    # 例如，将第一个通道的第一个像素点设置为红色（1.0, 0.0, 0.0）
    tensor[0, 0, 0, 0] = 1.0  # 红色通道
    tensor[0, 1, 0, 0] = 0.0  # 绿色通道
    tensor[0, 2, 0, 0] = 0.0  # 蓝色通道

    # 如果你想要将张量中的所有像素点都设置为相同的颜色，可以使用以下代码
    # tensor[0, 0, :, :] = 1.0  # 将红色通道的所有像素点设置为1.0
    # tensor[0, 1, :, :] = 0.0  # 将绿色通道的所有像素点设置为0.0
    # tensor[0, 2, :, :] = 0.0  # 将蓝色通道的所有像素点设置为0.0

def test_2d():
    arr_2_d = np.asarray([[1, 2], [3, 4]], dtype=np.float16)
    print(arr_2_d.dtype)

if __name__ == '__main__':
    print_mac_info()
    clare_pic_numpy()
    test_2d()