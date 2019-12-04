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

class EDANet(nn.Module):

    def __init__(self, num_classes = 2, init_weights = True):
        super(EDANet, self).__init__()
        self.num_classes = num_classes

        self.model = torch.nn.ModuleDict({
            'downsample_1': DownSamplingBlock(3, 15),
            'downsample_2': DownSamplingBlock(15, 60),
            'EDABlock_1': EDABlock(60, 1),
            'EDABlock_2': EDABlock(100, 1),
            'EDABlock_3': EDABlock(240, 1),
            'EDABlock_4': EDABlock(520, 2),
            'EDABlock_5': EDABlock(220, 2),
            'downsample_3': DownSamplingBlock(260, 130),
            'EDABlock_6': EDABlock(130, 2),
            'EDABlock_7': EDABlock(170, 2),
            'EDABlock_8': EDABlock(210, 4),
            'EDABlock_9': EDABlock(250, 4),
            'EDABlock_10': EDABlock(290, 8),
            'EDABlock_11': EDABlock(330, 8),
            'EDABlock_12': EDABlock(370, 16),
            'EDABlock_13': EDABlock(410, 16),
            'Projection': nn.Conv2d(450, self.num_classes, kernel_size=1)
        })

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

    def forward(self, input, target):

        output = self.model['downsample_1'](input)
        output = self.model['downsample_2'](output)
        EDABlock_1_output = self.model['EDABlock_1'](output)
        EDABlock_2_output = self.model['EDABlock_2'](EDABlock_1_output)
        EDABlock_3_output = self.model['EDABlock_3'](torch.cat([EDABlock_1_output, EDABlock_2_output], dim=1))
        EDABlock_4_output = self.model['EDABlock_4'](torch.cat([EDABlock_1_output, EDABlock_2_output, EDABlock_3_output], dim=1))
        EDABlock_5_output = self.model['EDABlock_4'](
            torch.cat([EDABlock_1_output, EDABlock_2_output, EDABlock_3_output, EDABlock_4_output], dim=1))
        output = self.model['downsample_2'](EDABlock_5_output)
        EDABlock_6_output = self.model['EDABlock_6'](output)
        EDABlock_7_output = self.model['EDABlock_7'](EDABlock_6_output)
        EDABlock_8_output = self.model['EDABlock_8'](
            torch.cat([EDABlock_6_output, EDABlock_7_output], dim=1))
        EDABlock_9_output = self.model['EDABlock_9'](
            torch.cat([EDABlock_6_output, EDABlock_7_output, EDABlock_8_output], dim=1))
        EDABlock_10_output = self.model['EDABlock_10'](
            torch.cat([EDABlock_6_output, EDABlock_7_output, EDABlock_8_output, EDABlock_9_output], dim=1))
        EDABlock_11_output = self.model['EDABlock_11'](
            torch.cat([EDABlock_6_output, EDABlock_7_output, EDABlock_8_output, EDABlock_9_output, EDABlock_10_output], dim=1))
        EDABlock_12_output = self.model['EDABlock_12'](
            torch.cat([EDABlock_6_output, EDABlock_7_output, EDABlock_8_output, EDABlock_9_output, EDABlock_10_output, EDABlock_11_output], dim=1))
        EDABlock_13_output = self.model['EDABlock_13'](
            torch.cat([EDABlock_6_output, EDABlock_7_output, EDABlock_8_output, EDABlock_9_output, EDABlock_10_output, EDABlock_11_output, EDABlock_12_output], dim=1))
        output = self.model['Projection'](EDABlock_13_output)

        # Bilinear interpolation x8
        output = F.interpolate(output, scale_factor = 8, mode = 'bilinear', align_corners = True)
        output = torch.nn.functional.softmax(output, dim = 1)
        # compute loss
        loss = FocalLoss(self.num_classes)(output, target)

        return loss, output

if __name__ == '__main__':

    # for the inference only mode
    net = EDANet()
    print(net)

    input = Variable(torch.randn(1, 3, 512, 1024))
    target = Variable(torch.randn(1, 1, 512, 1024))
    output = net(input, target)
    print(output.size())