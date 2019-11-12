import torch
import torch.nn as nn
import torch.nn.functional as F
from DCNv2.dcn_v2 import DCN

BN_MOMENTUM = 0.1


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, planes):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE', d=1):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = d
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False, dilation=d),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.conv1 = DCN(in_channels, out_channels,
                         kernel_size=(3, 3), stride=1,
                         padding=1, dilation=1, deformable_groups=1)
        # self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4,
                                        padding=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x)


class Tinynet(nn.Module):
    def __init__(self, heads):

        super(Tinynet, self).__init__()
        self.heads = heads
        self.threshold = 0.3
        stage1 = [[3, 96, 32, False, 'RE', 1],
                  [3, 96, 32, False, 'RE', 1],
                  [3, 96, 32, False, 'RE', 1]]

        stage2 = [[3, 192, 64, False, 'RE', 2],
                  [3, 192, 64, False, 'RE', 1],
                  [3, 192, 64, False, 'RE', 1],
                  [3, 192, 64, False, 'RE', 1]]

        stage3 = [[3, 384, 128, True, 'RE', 2],
                  [3, 384, 128, True, 'RE', 1],
                  [3, 384, 128, True, 'RE', 1]]

        stage4 = [[3, 512, 256, False, 'HS', 2],
                  [3, 512, 256, False, 'HS', 1],
                  [3, 512, 256, False, 'HS', 1],
                  [3, 512, 256, False, 'HS', 1],
                  [3, 512, 256, False, 'HS', 1],
                  [3, 512, 256, False, 'HS', 1],
                  [3, 512, 256, False, 'HS', 1]]

        stage5 = [[3, 512, 256, True, 'HS', 1],
                  [3, 512, 256, True, 'HS', 1],
                  [3, 512, 256, True, 'HS', 1],
                  [3, 512, 256, True, 'HS', 1],
                  [3, 512, 256, True, 'HS', 1]]

        input_channel = 32
        self.features0 = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.features1 = []
        self.features2 = []
        self.features3 = []
        self.features4 = []
        self.features5 = []

        for k, exp, c, se, nl, s in stage1:
            output_channel = c
            exp_channel = exp
            self.features1.append(Bottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel
        for k, exp, c, se, nl, s in stage2:
            output_channel = c
            exp_channel = exp
            self.features2.append(Bottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel
        for k, exp, c, se, nl, s in stage3:
            output_channel = c
            exp_channel = exp
            self.features3.append(Bottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel
        for k, exp, c, se, nl, s in stage4:
            output_channel = c
            exp_channel = exp
            self.features4.append(Bottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel
        for k, exp, c, se, nl, s in stage5:
            output_channel = c
            exp_channel = exp
            self.features5.append(Bottleneck(input_channel, output_channel, k, s, exp_channel, se, nl, d=2))
            input_channel = output_channel
        self.features0 = nn.Sequential(*self.features0)
        self.features1 = nn.Sequential(*self.features1)
        self.features2 = nn.Sequential(*self.features2)
        self.features3 = nn.Sequential(*self.features3)
        self.features4 = nn.Sequential(*self.features4)
        self.features5 = nn.Sequential(*self.features5)
        self.upstage1 = Upsample(256, 128)
        self.upstage2 = Upsample(128, 64)

        for head in self.heads:
            num_classes = self.heads[head]
            if head == 'hm':
                fc = nn.Sequential(
                    nn.Conv2d(64, 64,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, num_classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True))
                fc[-1].bias.data.fill_(-2.19)
            else:
                fc = nn.Sequential(
                    nn.Conv2d(64, 256,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, num_classes,
                              kernel_size=1, stride=1))

            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.features0(x)
        x = self.features1(x)
        x = self.features2(x)
        x1 = self.features3(x)
        x2 = self.features4(x1)
        x3 = self.features5(x2)
        x3 = self.upstage1(x3) + x1
        x3 = self.upstage2(x3) + x
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x3)
        return z


def tinynet():
    return Tinynet({'hm': 1, 'corner': 4})


if __name__ == '__main__':
    net = tinynet().cuda()
    a = torch.randn(1, 3, 512, 512).cuda()
    b = net(a)
    for i in b:
        print(i, b[i].shape)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    input_size = torch.randn(3, 512, 512)
    # # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
    from thop import profile

    flops, params = profile(net, inputs=(a,))
    print(flops)
    print(params)
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fB' % (flops / 1000000000.0))
