# import torch
# import torch.nn as nn
#
# class DEPTHWISECONV(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DEPTHWISECONV, self).__init__()
#         # 也相当于分组为1的分组卷积
#         self.depth_conv = nn.Conv2d(in_channels=in_ch,
#                                     out_channels=in_ch,
#                                     kernel_size=3,
#                                     stride=1,
#                                     padding=1,
#                                     groups=in_ch)
#         self.point_conv = nn.Conv2d(in_channels=in_ch,
#                                     out_channels=out_ch,
#                                     kernel_size=1,
#                                     stride=1,
#                                     padding=0,
#                                     groups=1)
#
#     def forward(self, input):
#         out = self.depth_conv(input)
#         out = self.point_conv(out)
#         return out
#
# # # 实例化 DEPTHWISECONV 类
# # in_channels = 1
# # out_channels = 1  # 你可以根据需要调整输出通道数
# # depthwise_conv = DEPTHWISECONV(in_channels=1, out_channels=1)
# #
# # # 创建一个形状为 (2, 1, 256, 256) 的随机张量
# # input_tensor = torch.randn(2, 1, 256, 256)
# #
# # # 将张量输入到 DEPTHWISECONV 实例中进行前向传播
# # output = depthwise_conv(input_tensor)
# #
# # # 打印输出张量的形状
# # print("输出张量的形状:", output.shape)
#

import torch
import torch.nn as nn

# 定义深度可分离卷积层
class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 线性归一化层
        self.normalization = nn.BatchNorm2d(1)
        # 1x1 卷积层
        self.conv1x1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1)
        # 3x3 卷积层
        self.conv3x3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 深度可分离卷积层
        self.depthwise_conv = DEPTHWISECONV(in_ch=64, out_ch=1)

    def forward(self, x):
        x = self.normalization(x)
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        x = self.depthwise_conv(x)
        return x

# 创建模型实例
model = Model()

# 创建一个 2x1x256x256 的随机张量
input_tensor = torch.randn(2, 1, 256, 256)

# 前向传播
output = model(input_tensor)

# 打印输出张量的形状
print("Output tensor shape:", output.shape)