import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, padding_mode="zeros"):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode),
            nn.LeakyReLU(0.1, inplace=False)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)
        )

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convd2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convsf1 = nn.Conv2d(3, 128, 7, padding=3)
        self.convsf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.conv = nn.Conv2d(64+128+192, 128-6, 3, padding=1)

    def forward(self, flow, disp, sf, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        d = F.relu(self.convd1(disp))
        d = F.relu(self.convd2(d))
        scene_f = F.relu(self.convsf1(sf))
        scene_f = F.relu(self.convsf2(scene_f))
        cor_flo = torch.cat([cor, flo, d, scene_f], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow, disp, sf], dim=1)

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)

        self.conv_sf = nn.Sequential(
            conv(128, 64),
            conv(64, 32),
            conv(32, 3, isReLU=False)
            )
        self.conv_d1 = nn.Sequential(
            conv(128, 64),
            conv(64, 32),
            conv(32, 1, isReLU=False)
            )
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, disp, sf):
        motion_features = self.encoder(flow, disp, sf, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        disp = self.conv_d1(net)
        sf = self.conv_sf(net)

        # scale mask to balence gradients
        mask = self.mask(net.detach())
        return net, mask, disp, sf
