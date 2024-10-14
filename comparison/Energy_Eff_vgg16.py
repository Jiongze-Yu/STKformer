import torch
import torch.nn as nn
import torchvision.models as models


class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=1000):  # 默认num_classes为4，可以自定义
        super(VGG16Classifier, self).__init__()

        # 加载预训练的VGG16模型
        self.vgg16 = models.vgg16(pretrained=True)

        # 修改分类头 (FC层)
        # 原VGG16的分类头是: Linear(4096 -> 4096) -> Linear(4096 -> 1000)
        # 我们替换为自定义的分类头以适应自定义的num_classes
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # VGG16的最后一层卷积层输出的是512个7x7特征图
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)  # 输出类别数为num_classes
        )

    def forward(self, x):
        x = self.vgg16(x)  # 使用修改后的VGG16网络进行前向传播
        return x


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):  # 默认num_classes为4，可以自定义
        super(VGG16, self).__init__()

        # VGG16特征提取部分（卷积层+池化层）
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
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸减半

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸减半

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸减半

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸减半
        )

        # VGG16的分类部分（全连接层）
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # VGG16的卷积层输出512个7x7的特征图
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),  # 自定义的类别数
        )

    def forward(self, x):
        # 前向传播
        x = self.features(x)  # 卷积层
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.classifier(x)  # 全连接层
        return x


if __name__ == '__main__':
    from thop import profile, clever_format
    # model = VGG16Classifier()
    model = VGG16()

    input = torch.rand([1, 3, 224, 224])  # [B, C, H, W]
    print('Image_input.shape = ', input.shape)

    output = model(input)
    # print(len(output))
    print('Model_output.shape = ', output.shape)

    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.2f")
    print(f"The number of parameters: {params}")
    print(f"number of GFLOPs: {flops}")
