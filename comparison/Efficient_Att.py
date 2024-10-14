import torch
import torch.nn as nn
import torch.nn.functional as F

# SE Block - 通道注意力模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)  # 全局平均池化
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Spatial Attention Module - 空间注意力模块
class SAModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

class SmoothLayer(nn.Module):
    def __init__(self, in_channels):
        super(SmoothLayer, self).__init__()
        self.smooth = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.smooth(x)

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# class Classifier(nn.Module):
#     def __init__(self, in_channels,k, num_classes):
#         super(Classifier, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(in_channels * k * k, 4096),  # VGG16的卷积层输出512个7x7的特征图
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),  # 自定义的类别数
#         )
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # 展平特征图
#         x = self.classifier(x)
#         return x

class VGG16AttentionFusion(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16AttentionFusion, self).__init__()

        # VGG16特征提取部分（只保留卷积层）
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸减半

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(128),  # 通道注意力
            SAModule(),  # 空间注意力
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸减半
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸减半
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸减半
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出尺寸减半
        )

        # 降维
        self.conv1x1_3 = Conv1x1(256, 256)  # 将X3降维到256个通道
        self.conv1x1_4 = Conv1x1(512, 256)  # 将X4降维到256个通道
        self.conv1x1_5 = Conv1x1(512, 256)  # 将X5降维到256个通道

        # 上采样模块
        self.upsample_4 = Upsample(scale_factor=2)  # 上采样到Block 4的大小
        self.upsample_3 = Upsample(scale_factor=2)  # 上采样到Block 3的大小

        # 平滑处理模块
        self.smooth_3 = SmoothLayer(256)
        self.smooth_4 = SmoothLayer(256)
        self.smooth_5 = SmoothLayer(256)

        # 三个分类器
        self.classifier_1 = Classifier(256, num_classes)
        self.classifier_2 = Classifier(256, num_classes)
        self.classifier_3 = Classifier(256, num_classes)

        # 分类器权重
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.w3 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # VGG提取特征
        x = self.features(x)  # 浅层特征X3
        x3 = self.block3(x)  # 中间层特征X4
        x4 = self.block4(x3)  # 深度特征X5
        x5 = self.block5(x4)

        # 降维处理
        x3 = self.conv1x1_3(x3)
        x4 = self.conv1x1_4(x4)
        x5 = self.conv1x1_5(x5)

        # 融合深层和中间层特征
        x4_upsampled = self.upsample_4(x5)  # 上采样到中间层大小
        xq = x4 + x4_upsampled

        # 融合浅层和中间层特征
        xq_upsampled = self.upsample_3(xq)  # 上采样到浅层大小
        xr = x3 + xq_upsampled

        # 输出特征平滑处理
        xsp = self.smooth_5(x5)  # 深度特征平滑
        xsq = self.smooth_4(xq)  # 中间特征平滑
        xsr = self.smooth_3(xr)  # 浅层特征平滑

        # print(xsp.shape, xsq.shape, xsr.shape)
        # 分类器输出
        score1 = self.classifier_1(xsp)
        score2 = self.classifier_2(xsq)
        score3 = self.classifier_3(xsr)

        # 加权融合分类器输出
        final_score = self.w1 * score1 + self.w2 * score2 + self.w3 * score3

        return final_score

# 实例化模型
# num_classes = 10  # 你可以自定义类别数
# model = VGG16AttentionFusion(num_classes=num_classes)
# print(model)
if __name__ == '__main__':
    from thop import profile, clever_format
    # model = VGG16Classifier()
    model = VGG16AttentionFusion()

    input = torch.rand([1, 3, 224, 224])  # [B, C, H, W]
    print('Image_input.shape = ', input.shape)

    output = model(input)
    # print(len(output))
    print('Model_output.shape = ', output.shape)

    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.2f")
    print(f"The number of parameters: {params}")
    print(f"number of GFLOPs: {flops}")