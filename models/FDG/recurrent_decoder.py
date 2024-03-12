"""
Depth decoder model (the Iterative Feature Aggregation Module) for FDG.

Parts of the code adapted from https://github.com/nianticlabs/manydepth.
Please refer to the license of the above repo.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ConvBlock, upsample, EnhancementAttentionModule


class DepthHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super(DepthHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        # return self.sig(self.conv2(self.relu(self.conv1(x))))
        return self.conv2(self.relu(self.conv1(x)))

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
    def __init__(self, hidden_dim=128, input_dim=128+256):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(input_dim, hidden_dim, (5,1), padding=(2,0))


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


class BasicEncoder(nn.Module):
    def __init__(self, in_e=128):
        super(BasicEncoder, self).__init__()
        self.convc1 = nn.Conv2d(in_e, in_e*2, 1, padding=0)
        self.convc2 = nn.Conv2d(in_e*2, int(in_e*1.5), 3, padding=1)

        self.convf1 = torch.nn.Sequential(
            nn.ReflectionPad2d(3),
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=0, bias=True),
            # torch.nn.LeakyReLU(inplace=True)
            )
        self.convf2 = torch.nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=0, bias=True),
            # torch.nn.LeakyReLU(inplace=True)
            )

        self.conv = nn.Conv2d(32+int(in_e*1.5), in_e-1, 3, padding=1)

    def forward(self, depth, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        dep = F.relu(self.convf1(depth))
        dep = F.relu(self.convf2(dep))

        cor_dep = torch.cat([cor, dep], dim=1)
        out = F.relu(self.conv(cor_dep))
        return torch.cat([out, depth], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, in_e=128, in_c=256):
        super(BasicUpdateBlock, self).__init__()

        self.encoder = BasicEncoder(in_e=in_e)
        self.gru = SepConvGRU(hidden_dim=in_e, input_dim=in_e+in_c)
        self.depth_head = DepthHead(in_e, hidden_dim=in_e)

        self.mask = nn.Sequential(
            nn.Conv2d(in_e, in_e * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_e * 2, 16 * 9, 1, padding=0))

    def forward(self, net, inp, corr, depth):
        depth_features = self.encoder(depth, corr)
        inp = torch.cat([inp, depth_features], dim=1)

        net = self.gru(net, inp)
        delta_depth = self.depth_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_depth




class RecurrentDecoder(nn.Module):
    def __init__(self, num_ch_enc=[64, 18, 36, 72, 144], iters=6):
        super(RecurrentDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.iters = iters
        # self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.update_block = BasicUpdateBlock(in_e=64, in_c=128)

        self.conv_e1 = EnhancementAttentionModule(self.num_ch_enc[-1], self.num_ch_enc[-2] * 2, 256)
        self.conv_e2 = EnhancementAttentionModule(256, self.num_ch_enc[-3] * 3, 128)
        self.conv_e3 = EnhancementAttentionModule(128, self.num_ch_enc[-4] * 3 + 64, 64)


        self.conv_c1 = EnhancementAttentionModule(self.num_ch_enc[-1], self.num_ch_enc[-2] * 2, 256)
        self.conv_c2 = EnhancementAttentionModule(256, self.num_ch_enc[-3] * 3, 128)
        self.conv_c3 = EnhancementAttentionModule(64, self.num_ch_enc[-4] * 3 + 64, 64, False)
        self.conv_c4 = ConvBlock(64, 64)

        self.sigmoid = nn.Sigmoid()

    def upsample_depth(self, depth, mask):
        """ Upsample depth map [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, C, H, W = depth.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_depth = nn.functional.unfold(depth, [3, 3], padding=1)
        up_depth = up_depth.view(N, C, 9, 1, 1, H, W)

        up_depth = torch.sum(mask * up_depth, dim=2)
        up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)
        return up_depth.reshape(N, C, 4 * H, 4 * W)

    def forward(self, encoder_fea, context_fea, test_mode=False, depth_init=None):
        self.outputs = {}

        e_fea = self.conv_e1(encoder_fea[-1], encoder_fea[-2])
        e_fea = self.conv_e2(e_fea, encoder_fea[-3])
        e_fea = self.conv_e3(e_fea, encoder_fea[-4])

        c_fea = self.conv_c1(context_fea[-1], context_fea[-2])
        c_fea = self.conv_c2(c_fea, context_fea[-3])

        net, inp = torch.split(c_fea, [c_fea.size()[1] // 2, c_fea.size()[1] // 2], dim=1)

        net = self.conv_c3(net, context_fea[-4])

        inp = self.conv_c4(inp)
        inp = upsample(inp)
        inp = torch.relu(inp)

        b, c, h, w = e_fea.size()

        if depth_init == None:
            depth = torch.zeros([b, 1, h, w], requires_grad=True).to(e_fea.device)
        else:
            depth = F.interpolate(depth_init, [h, w], mode="nearest")

        for itr in range(self.iters):

            net, up_mask, delta_depth = self.update_block(net, inp, e_fea, depth)

            depth = self.sigmoid(depth + delta_depth)

            # upsample predictions
            depth_up = self.upsample_depth(depth, up_mask)

            self.outputs[("disp", 0, itr)] = depth_up


        if test_mode:
            return depth_up, depth, up_mask
        else:
            return self.outputs

