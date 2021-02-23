import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from util.cc_attention import CrissCrossAttention

class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3GNReLU, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.GroupNorm(16,out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block1(x)
        return x


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


def CC_module(proj_query, proj_key, proj_value):

    m_batchsize, _, height, width = proj_value.size()

    proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous(). \
        view(m_batchsize * width, -1, height).permute(0, 2, 1)
    proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous(). \
        view(m_batchsize * height, -1, width).permute(0, 2, 1)

    proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
    proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
    proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
    proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

    A1 = proj_query_H / (torch.sqrt(torch.sum(torch.mul(proj_query_H, proj_query_H), dim=-1, keepdim=True))+1e-10)
    B1 = proj_key_H / (torch.sqrt(torch.sum(torch.mul(proj_key_H, proj_key_H), dim=1, keepdim=True))+1e-10)
    energy_H = torch.bmm(A1, B1).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)

    A2 = proj_query_W / (torch.sqrt(torch.sum(torch.mul(proj_query_W, proj_query_W), dim=-1, keepdim=True))+1e-10)
    B2 = proj_key_W / (torch.sqrt(torch.sum(torch.mul(proj_key_W, proj_key_W), dim=1, keepdim=True))+1e-10)
    energy_W = torch.bmm(A2, B2).view(m_batchsize, height, width, width)
    concate = F.softmax(torch.cat([energy_H, energy_W], 3), 3)

    att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
    att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
    out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
    out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
    return out_H + out_W

def grid_PA(cont_p, query, key, value, bin, START_H, END_H, START_W, END_W):

    for i in range(0, bin):
        for j in range(0, bin):
            cont_p=cont_p.clone()
            value = value.clone()
            key = key.clone()
            query = query.clone()

            value_local = value[:, :, START_H[i, j]:END_H[i, j], START_W[i, j]:END_W[i, j]]

            #print(i,j, START_H[i, j],END_H[i, j], START_W[i, j], END_W[i, j])
            #value_local.backward(torch.ones(1, 64, 256 // bin, 108 // bin).cuda(), retain_graph=True) # backword error checking

            query_local = query[:, :, START_H[i, j]:END_H[i, j], START_W[i, j]:END_W[i, j]]
            key_local = key[:, :, START_H[i, j]:END_H[i, j], START_W[i, j]:END_W[i, j]]

            cont_p_local = CC_module(query_local, key_local, value_local)

            cont_p[:, :, START_H[i, j]:END_H[i, j], START_W[i, j]:END_W[i, j]] = cont_p_local

    return cont_p

class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.contiguous().view(m_batchsize, C, -1)  # B,C,-1
        proj_key = x.contiguous().view(m_batchsize, C, -1).permute(0, 2, 1)  # B,-1,C

        A1 = proj_query / (torch.sqrt(torch.sum(torch.mul(proj_query, proj_query), dim=-1, keepdim=True))+1e-10)
        B1 = proj_key / (torch.sqrt(torch.sum(torch.mul(proj_key, proj_key), dim=1, keepdim=True))+1e-10)

        energy = torch.bmm(A1, B1)  # B,C,C

        attention = self.softmax(energy)
        proj_value = x.contiguous().view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class SAM_Module(nn.Module):
    """ Scale attention module"""

    def __init__(self, num, in_dim):
        super(SAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.ModuleList()
        self.key_conv = nn.ModuleList()
        self.value_conv = nn.ModuleList()
        for i in range(num):
            self.query_conv.append(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1))
            self.key_conv.append(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1))
            self.value_conv.append(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, L):
        """
            inputs :
                x : input feature maps(B,S,C,H,W)
            returns :
                out : attention value + input feature
                attention: B X (CHW) X (CHW)
        """
        x = torch.stack(L, dim=1)
        m_batchsize, S, C, height, width = x.size()
        qs = []
        ks = []

        for i in range(len(L)):
            qs.append(self.query_conv[i](L[i]))  # B,c,H,W
            ks.append(self.key_conv[i](L[i]))
        proj_query = torch.stack(qs, dim=1).view(m_batchsize, S, -1)  # B,S,c,H,W --> B,S,-1
        proj_key = torch.stack(ks, dim=1).view(m_batchsize, S, -1).permute(0, 2, 1)  # B,S,c,H,W --> B,-1,S

        A1 = proj_query / (torch.sqrt(torch.sum(torch.mul(proj_query, proj_query), dim=-1, keepdim=True))+1e-10)
        B1 = proj_key / (torch.sqrt(torch.sum(torch.mul(proj_key, proj_key), dim=1, keepdim=True))+1e-10)

        energy = torch.bmm(A1, B1)  # B,S,S
        attention = self.softmax(energy)

        vs = []
        for i in range(len(L)):
            vs.append(self.value_conv[i](L[i]))  # B,C,H,W
        proj_value = torch.stack(vs, dim=1).view(m_batchsize, S, -1)  # B,S,CHW

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, S, C, height, width)

        out = self.gamma * out + x

        re = []
        for i in range(len(L)):
            re.append(out[:, i, :, :, :])
        return re

class FuCont_S(nn.Module):
    def __init__(self, num, in_channel):
        super(FuCont_S, self).__init__()
        self.SA = SAM_Module(num, in_channel)
    def forward(self, L):
        L=self.SA(L)
        return L

## PSP
class FuCont_PSP_PC(nn.Module):
    def __init__(self, args, in_dim, bin):
        super(FuCont_PSP_PC, self).__init__()
        self.args = args
        self.bin = bin
        flag = 0
        if 'p' in self.args.mode:
            flag += 1
            self.query_conv_p = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
            self.key_conv_p = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
            self.value_conv_p = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        if 'c' in self.args.mode:
            flag += 1
            self.CA = CAM_Module(in_dim)
        if flag == 0:
            raise NotImplementedError

    def forward(self, x):

        _, _, h, w = x.size()

        step_h, step_w = h // self.bin, w // self.bin

        START_H = np.zeros([self.bin, self.bin]).astype(int)
        END_H = np.zeros([self.bin, self.bin]).astype(int)
        START_W = np.zeros([self.bin, self.bin]).astype(int)
        END_W = np.zeros([self.bin, self.bin]).astype(int)

        for i in range(0, self.bin):
            for j in range(0, self.bin):
                start_h, start_w = i * step_h, j * step_w
                end_h, end_w = min(start_h + step_h, h), min(start_w + step_w, w)
                if i == (self.bin - 1):
                    end_h = h
                if j == (self.bin - 1):
                    end_w = w
                START_H[i, j] = start_h
                END_H[i, j] = end_h
                START_W[i, j] = start_w
                END_W[i, j] = end_w

        if self.args.mode == 'p_c_s':

            cont_p = torch.zeros(x.shape).cuda()

            for cnt in range(2):

                #print('bin, cnt', self.bin, cnt)

                value = self.value_conv_p(x)
                query = self.query_conv_p(x)
                key = self.key_conv_p(x)

                cont_p = grid_PA(cont_p, query, key, value, self.bin, START_H, END_H, START_W, END_W)

                x = cont_p  # recurrent

            x = self.CA(x)

        elif self.args.mode == 'c_p_s':

            x = self.CA(x)

            cont_p = torch.zeros(x.shape).cuda()

            for cnt in range(2):

                value = self.value_conv_p(x)
                query = self.query_conv_p(x)
                key = self.key_conv_p(x)

                cont_p = grid_PA(cont_p, query, key, value, self.bin, START_H, END_H, START_W, END_W)

                x = cont_p  # recurrent

        elif self.args.mode == 'p+c_s':

            x1 = self.CA(x)

            x2 = x

            cont_p = torch.zeros(x2.shape).cuda()

            for cnt in range(2):

                value = self.value_conv_p(x2)
                query = self.query_conv_p(x2)
                key = self.key_conv_p(x2)

                cont_p = grid_PA(cont_p, query, key, value, self.bin, START_H, END_H, START_W, END_W)

                x2 = cont_p  # recurrent

            x = x1 + x2

        elif self.args.mode == 'p_s_c':

            cont_p = torch.zeros(x.shape).cuda()

            for cnt in range(2):

                value = self.value_conv_p(x)
                query = self.query_conv_p(x)
                key = self.key_conv_p(x)

                cont_p = grid_PA(cont_p, query, key, value, self.bin, START_H, END_H, START_W, END_W)

                x = cont_p  # recurrent
        else:
            raise NotImplementedError

        return x

class PPM_FC(nn.Module):
    def __init__(self, args, in_dim, reduction_dim, bins):
        super(PPM_FC, self).__init__()
        num = len(bins)
        self.args = args

        self.conv1=nn.Conv2d(in_dim,reduction_dim,1, bias=False)
        self.fuc_pc = nn.ModuleList()
        for bin in bins:
            self.fuc_pc.append(FuCont_PSP_PC(args, reduction_dim, bin))
        self.fuc_s = FuCont_S(num, reduction_dim)
        if self.args.mode == 'p_s_c':
            self.CA = CAM_Module(4*reduction_dim)

        self.conv = nn.Conv2d(in_dim+256, 2*reduction_dim, 1, bias=False)
        self.gn = nn.GroupNorm(16,2*reduction_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        out = [x]
        L = []
        x = self.conv1(x)  # dim reduction
        for path in self.fuc_pc:
            L.append(path(x))
        L = self.fuc_s(L)
        if self.args.mode == 'p_s_c':
            L=torch.cat(L,1)
            L = self.CA(L)
            out=[res,L]
        else:
            out.extend(L)

        out = torch.cat(out, 1)
        out = self.conv(out)
        out = self.gn(out)
        out = self.relu(out)
        return out

## ASPP
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.gn = nn.GroupNorm(16,planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.gn(x)
        return self.relu(x)


class FuCont_ASPP_PC(nn.Module):
    def __init__(self, args, inplanes, dilations):
        super(FuCont_ASPP_PC, self).__init__()
        self.args=args

        flag = 0

        self.aspp1 = _ASPPModule(int(inplanes / 4), int(inplanes / 4), 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(int(inplanes / 4), int(inplanes / 4), 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(int(inplanes / 4), int(inplanes / 4), 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(int(inplanes / 4), int(inplanes / 4), 3, padding=dilations[3], dilation=dilations[3])
        self.conv1 = Conv3x3GNReLU(int(inplanes / 4), int(inplanes / 4))

        if 'p' in self.args.mode:
            flag += 1
            in_dim = int(inplanes / 4)
            self.query_conv_p = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
            self.key_conv_p = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
            self.value_conv_p = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        if 'c' in self.args.mode[:-1]:
            flag += 1
            in_dim = int(inplanes / 4)
            self.CA=nn.ModuleList()
            for i in range(5):
                self.CA.append(CAM_Module(in_dim))
        if flag == 0:
            raise NotImplementedError
    def forward(self, x):

        if self.args.mode == 'p_c_s':

            x1 = self.aspp1(x)
            x2 = self.aspp2(x)
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)
            x5 = self.conv1(x)

            for cnt in range(2):
                value = self.value_conv_p(x5)
                query = self.query_conv_p(x5)
                key = self.key_conv_p(x5)

                cont_p = CC_module(query, key, value)

                x5 = cont_p  # recurrent

            L=[x1,x2,x3,x4,x5]

            for i in range(5):
                L[i]=self.CA[i](L[i])

        elif self.args.mode == 'c_p_s':

            L=[]

            for i in range(5):
                L.append(self.CA[i](x))

            L[0] = self.aspp1(L[0])
            L[1] = self.aspp2(L[1])
            L[2] = self.aspp3(L[2])
            L[3] = self.aspp4(L[3])
            L[4] = self.conv1(L[4])

            for cnt in range(2):
                value = self.value_conv_p(L[4])
                query = self.query_conv_p(L[4])
                key = self.key_conv_p(L[4])

                cont_p = CC_module(query, key, value)

                L[4] = cont_p  # recurrent

        elif self.args.mode == 'p+c_s':

            x1 = self.aspp1(x)
            x2 = self.aspp2(x)
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)
            x5 = self.conv1(x)

            for cnt in range(2):
                value = self.value_conv_p(x5)
                query = self.query_conv_p(x5)
                key = self.key_conv_p(x5)

                cont_p = CC_module(query, key, value)

                x5 = cont_p  # recurrent

            L = [x1, x2, x3, x4, x5]

            for i in range(5):
                L[i]=L[i]+self.CA[i](x)

        elif self.args.mode == 'p_s_c':

            x1 = self.aspp1(x)
            x2 = self.aspp2(x)
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)
            x5 = self.conv1(x)

            for cnt in range(2):
                value = self.value_conv_p(x5)
                query = self.query_conv_p(x5)
                key = self.key_conv_p(x5)

                cont_p = CC_module(query, key, value)

                x5 = cont_p  # recurrent

            L = [x1, x2, x3, x4, x5]

        else:
            raise NotImplementedError

        return L



class ASPP_FC(nn.Module):
    def __init__(self, args, inplanes, dilations):
        super(ASPP_FC, self).__init__()
        self.args = args

        self.conv1 = nn.Conv2d(inplanes, 64, 1, bias=False)
        self.fuc_pc = FuCont_ASPP_PC(args, 256, dilations)
        self.fuc_s = FuCont_S(5, 64)
        if self.args.mode == 'p_s_c':
            self.CA = CAM_Module(64*5)

        self.conv = nn.Conv2d(64 * 5, 64 * 2, 1, bias=False)
        self.gn = nn.GroupNorm(16, 64 * 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        L = self.fuc_pc(x)
        L = self.fuc_s(L)
        if self.args.mode == 'p_s_c':
            L = torch.cat(L, 1)
            out = self.CA(L)
        else:
            out = torch.cat(L, dim=1)

        out = self.conv(out)
        out = self.gn(out)
        out = self.relu(out)
        return out


class fucontnet(nn.Module):
    def __init__(self, args, spec_band, num_classes, init_weights=True, inner_channel=256):
        super(fucontnet, self).__init__()

        self.args = args

        self.encoder = nn.Sequential(
            Conv3x3GNReLU(spec_band, 64),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            Conv3x3GNReLU(64, 112),
            Conv3x3GNReLU(112, 160),
            Conv3x3GNReLU(160, 208),
            Conv3x3GNReLU(208, 256),

        )

        if args.head == 'psp':
            bins = (1, 2, 3, 6)
            self.head = PPM_FC(args, inner_channel, 64, bins)
        elif args.head == 'aspp':
            max_d = int(((min(args.input_size) / 2) * 3 / 4 + 1 - 1) / 2)
            dilations = (1, max_d // 4, max_d // 2, max_d)
            self.head = ASPP_FC(args, inner_channel, dilations)
        else:
            raise NotImplementedError
        fea_dim = 128 #int(inner_channel/2)
        self.cls = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(fea_dim, num_classes, kernel_size=1, bias=False)
            )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        _, _, h, w = x.size()

        x = self.encoder(x)
        x = self.head(x)
        x = self.cls(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input = torch.rand(2, 103, 256, 256).cuda()
    parser = argparse.ArgumentParser(description="test")
    args = parser.parse_args()
    args.network = 'FContNet'
    args.head = 'aspp'
    args.mode = 'p_s_c'
    args.input_size = [256,256]
    args.network = 'FContNet'
    print('Implementing FcontNet in {} mode with {} head!'.format(args.mode,args.head))

    model = fucontnet(args, 103, 9).cuda()
    model.eval()
    # print(model)
    output = model(input)
    print('fucontnet', output.size())
