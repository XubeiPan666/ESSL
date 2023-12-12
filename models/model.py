import torch
import torch.nn as nn
import torch.nn.functional as F


class AM(nn.Module):
    def __init__(self, channel=32, divide=4):
        super(AM, self).__init__()
        self.am = nn.Sequential(
            nn.Linear(channel, channel // divide),
            nn.ReLU(inplace=True),
            nn.Linear(channel // divide, channel),
            nn.Sigmoid()
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.am(w).view(b, c, 1, 1, 1)
        return w * x


class RAB(nn.Module):
    def __init__(self, input_channel=32, output_channel=64, divide=4, time_length=1, strides=1, use_1x1conv=False):
        super(RAB, self).__init__()
        self.time_length = time_length
        self.strides = strides
        self.use_1x1conv = use_1x1conv
        self.AM = AM(channel=output_channel)

        self.conv1x1 = nn.Conv3d(input_channel, output_channel, kernel_size=(1, 1, 1),
                                 stride=(1, self.strides, self.strides), bias=False)

        self.model = nn.Sequential(
            nn.Conv3d(input_channel, output_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                      bias=False),
            nn.InstanceNorm3d(output_channel),
            nn.ReLU(),
            nn.Conv3d(output_channel, output_channel, kernel_size=(3, 3, 3), stride=(1, self.strides, self.strides),
                      padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(output_channel),
        )

    def forward(self, x):
        if self.use_1x1conv:
            y1 = self.conv1x1(x)
        else:
            y1 = x
        y2 = self.model(x)
        y2 = self.AM(y2)
        y = y1 + y2
        y = F.relu(y)
        return y


class RANet(nn.Module):
    def __init__(self, time_length=7, num_classes=None, use_1x1conv=False):
        super(RANet, self).__init__()
        if num_classes is None:
            num_classes = [49, 81]
        self.num_classes = num_classes
        self.use_1x1conv = use_1x1conv
        self.model1 = nn.Sequential(
            RAB(input_channel=3, output_channel=32, strides=2, use_1x1conv=True),
            RAB(input_channel=32, output_channel=32, strides=1, use_1x1conv=self.use_1x1conv),

            RAB(input_channel=32, output_channel=64, strides=2, use_1x1conv=True),
            RAB(input_channel=64, output_channel=64, strides=1, use_1x1conv=self.use_1x1conv),

            RAB(input_channel=64, output_channel=128, strides=2, use_1x1conv=True),
            RAB(input_channel=128, output_channel=128, strides=1, use_1x1conv=self.use_1x1conv),

            RAB(input_channel=128, output_channel=256, strides=2, use_1x1conv=True),
            RAB(input_channel=256, output_channel=256, strides=1, use_1x1conv=self.use_1x1conv),

            nn.AvgPool3d(kernel_size=(time_length, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        )

        self.model2 = nn.Sequential(
            RAB(input_channel=3, output_channel=32, strides=2, use_1x1conv=True),
            RAB(input_channel=32, output_channel=32, strides=1, use_1x1conv=self.use_1x1conv),

            RAB(input_channel=32, output_channel=64, strides=2, use_1x1conv=True),
            RAB(input_channel=64, output_channel=64, strides=1, use_1x1conv=self.use_1x1conv),

            RAB(input_channel=64, output_channel=128, strides=2, use_1x1conv=True),
            RAB(input_channel=128, output_channel=128, strides=1, use_1x1conv=self.use_1x1conv),

            RAB(input_channel=128, output_channel=256, strides=2, use_1x1conv=True),
            RAB(input_channel=256, output_channel=256, strides=1, use_1x1conv=self.use_1x1conv),

            nn.AvgPool3d(kernel_size=(time_length, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        )

        self.classifier_1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes[0])
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes[1])
        )

    def forward(self, x):
        out1 = self.model1(x)
        out1 = out1.view((out1.size(0), -1))
        out1 = self.classifier_1(out1)

        out2 = self.model1(x)
        out2 = out2.view((out2.size(0), -1))
        out2 = self.classifier_2(out2)
        return out1, out2


class WideBranchNet(nn.Module):
    def __init__(self, time_length=7, num_classes=[127, 8]):
        super(WideBranchNet, self).__init__()

        self.time_length = time_length
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),

            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(self.time_length, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
        )
        self.max2d = nn.MaxPool2d(2, 2)
        self.classifier_1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes[0])
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes[1])
        )

    def forward(self, x):
        out = self.model(x)
        out = out.squeeze(2)
        out = self.max2d(self.conv2d(out))
        out = out.view((out.size(0), -1))
        out1 = self.classifier_1(out)
        out2 = self.classifier_2(out)
        return out1, out2



