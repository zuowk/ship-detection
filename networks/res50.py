import torch
from torch import nn
from DCNv2.dcn_v2 import DCN


def conv_batch(in_num, out_num, kernel_size=3, stride=1, d=1):
    padding = d * ((kernel_size - 1) // 2)
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride,
                  padding=padding, bias=False, dilation=d),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


class Dilation_block(nn.Module):
    def __init__(self, in_num, out_num, kernel_size=3):
        super(Dilation_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_num, out_num, kernel_size=kernel_size,
                                            stride=1, dilation=2, padding=2, bias=False),
                                  nn.BatchNorm2d(out_num),
                                  nn.LeakyReLU())
        self.res = nn.Conv2d(in_num, out_num, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x) + self.res(x)


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels, d=1):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels / 2)

        self.layer1 = conv_batch(in_channels, reduced_channels,
                                 kernel_size=1)
        self.layer2 = conv_batch(reduced_channels, in_channels, d=d)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet(nn.Module):
    def __init__(self, block):
        super(Darknet, self).__init__()

        self.heads = ['hm', 'corner']
        self.threshold = 0.23

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = Dilation_block(512, 640)
        self.residual_block5 = self.make_layer(block, in_channels=640, num_blocks=4, d=2)

        self.deconv_layers = self._make_deconv_layer(
            2,
            [256, 128],
            [4, 4],
        )

        for head in self.heads:
            if head == 'hm':
                fc = nn.Sequential(
                    nn.Conv2d(128, 256,
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 1,
                              kernel_size=1, stride=1,
                              padding=0, bias=True))
                fc[-1].bias.data.fill_(-2.19)
            else:
                fc = nn.Sequential(
                    nn.Conv2d(128, 256,
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 4,
                              kernel_size=1, stride=1,
                              padding=0, bias=True),
                    nn.BatchNorm2d(4))
            self.__setattr__(head, fc)

    def make_layer(self, block, in_channels, num_blocks, d=1):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, d=d))
        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        inplanes = 640
        for i in range(num_layers):
            kernel, padding, output_padding = 4, 1, 0

            planes = num_filters[i]
            fc = DCN(inplanes, planes,
                     kernel_size=(3, 3), stride=1,
                     padding=1, dilation=1, deformable_groups=1)
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.deconv_layers(out)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(out)

        return ret


def darknet():
    return Darknet(DarkResidualBlock)


if __name__ == '__main__':
    net = darknet().cuda()
    a = torch.randn(1, 3, 512, 512).cuda()
    b = net(a)
    # print(b.shape)
    for i in b:
        print(i, b[i].shape)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    input_size = torch.randn(3, 512, 512)
    from thop import profile

    flops, params = profile(net, inputs=(a,))
    print(flops)
    print(params)
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fB' % (flops / 1000000000.0))
