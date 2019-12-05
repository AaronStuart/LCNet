import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from loss.focal_loss import FocalLoss


class DownSamplingBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DownSamplingBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        if self.in_channel > self.out_channel:
            self.conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size = 3, stride = 2, padding = 1)
        else:
            self.conv = nn.Conv2d(self.in_channel, self.out_channel - self.in_channel, kernel_size = 3, stride = 2, padding = 1)
            self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        if self.in_channel > self.out_channel:
            output = self.conv(x)
        else:
            output = torch.cat([self.conv(x), self.pool(x)], dim = 1)

        return F.relu(self.bn(output))

class EDABlock(nn.Module):

    def __init__(self, in_channel, short_cut = None, dilation_rate = 1, k = 40, dropprob = 0.02):
        super(EDABlock, self).__init__()
        self.in_channel = in_channel
        self.short_cut = short_cut
        self.dilation_rate = dilation_rate
        self.k = k
        self.dropprob = dropprob

        self.conv1x1 = nn.Conv2d(in_channel, k, kernel_size = 1)
        self.bn0 = nn.BatchNorm2d(k)

        self.conv3x1_1 = nn.Conv2d(k, k, kernel_size = (3, 1), padding = (1, 0))
        self.conv1x3_1 = nn.Conv2d(k, k, kernel_size = (1, 3), padding = (0, 1))
        self.bn1 = nn.BatchNorm2d(k)

        self.conv3x1_2 = nn.Conv2d(k, k, kernel_size=(3, 1), padding=(dilation_rate, 0), dilation = dilation_rate)
        self.conv1x3_2 = nn.Conv2d(k, k, kernel_size=(1, 3), padding=(0, dilation_rate), dilation = dilation_rate)
        self.bn2 = nn.BatchNorm2d(k)
        
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        output = self.conv1x1(x)
        output = self.bn0(output)
        output = F.relu(output)

        output = self.conv3x1_1(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        output = F.relu(output)

        if self.dropprob > 0:
            output = self.dropout(output)

        return torch.cat([output, x ], dim = 1)

class ProjectionLayer(nn.Module):

    def __init__(self, input_channels, num_classes):
        super(ProjectionLayer, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.conv = nn.Conv2d(input_channels, num_classes, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)

class EDANet(nn.Module):

    def __init__(self, num_classes, init_weights = True, device = torch.device("cpu")):
        super(EDANet, self).__init__()
        self.num_classes = num_classes
        self.device = device

        self.model = nn.Sequential(
            DownSamplingBlock(3, 15),
            DownSamplingBlock(15, 60),
            EDABlock(60, 1),
            EDABlock(100, 1),
            EDABlock(140, 1),
            EDABlock(180, 2),
            EDABlock(220, 2),
            DownSamplingBlock(260, 130),
            EDABlock(130, 2),
            EDABlock(170, 2),
            EDABlock(210, 4),
            EDABlock(250, 4),
            EDABlock(290, 8),
            EDABlock(330, 8),
            EDABlock(370, 16),
            EDABlock(410, 16),
            ProjectionLayer(450, self.num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, target = None):
        if self.training and target is None:
            raise ValueError("In training mode, targets should be passed")

        output = self.model(input)
        # Bilinear interpolation x8
        output = F.interpolate(output, scale_factor = 8, mode = 'bilinear', align_corners = True)
        # Softmax
        output = torch.nn.functional.softmax(output, dim=1)

        return output

if __name__ == '__main__':

    net = EDANet(num_classes=2)
    input = Variable(torch.randn(1, 3, 512, 1024))
    label = Variable(torch.zeros(1, 1, 512, 1024))

    # train mode
    net.train()
    output = net(input, label)
    loss = FocalLoss(net.num_classes)(output, label)
    print("train_loss:", loss)

    # inference mode
    net.eval()
    output = net(input)
    print("output_size:", output.size())


