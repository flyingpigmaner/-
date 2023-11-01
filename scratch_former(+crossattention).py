import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from functools import partial
import torch.nn.functional as F
import numpy as np
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math

from models.encoder import EncoderBlock, CEFF, LayerNorm, BasicConv2d
from mmseg.visualization import SegLocalVisualizer
import mmcv

def visualize(img, feature,out_file,imgname):
    seg_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend'),
                      dict(type='TensorboardVisBackend'),
                      dict(type='WandbVisBackend')],
        save_dir='temp_dir',
        alpha=0.5)
    image = mmcv.imread(img, 'color')
    # add feature map to wandb visualizer
    drawn_img = seg_visualizer.draw_featmap(feature, image, channel_reduction='squeeze_mean', )  # 'select_max''squeeze_mean' topk=6, arrangement=(2, 3)
        # seg_visualizer.show(drawn_img)
    mmcv.imwrite(mmcv.bgr2rgb(drawn_img), out_file + '/' + f'{imgname}.png')

def tensor2im(input_image, imtype=np.uint8):
    """"
    Parameters:
        input_image (tensor) --  输入的tensor，维度为CHW，注意这里没有batch size的维度
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.5, 0.5, 0.5]  # dataLoader中设置的mean参数，需要从dataloader中拷贝过来
    std = [0.5, 0.5, 0.5]  # dataLoader中设置的std参数，需要从dataloader中拷贝过来
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # 如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):  # 反标准化，乘以方差，加上均值
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255  # 反ToTensor(),从[0,1]转为[0,255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
    return image_numpy.astype(imtype)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))

            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)

        _, _, H, W = x.shape
        #x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)

            if output_h > input_h or output_w > output_h:

                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):

                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


# Transformer Decoder MLP
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


#Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

#Intermediate prediction module
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.PReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)


# Channel-wise Correlation
class CCorrM(nn.Module):
    def __init__(self, all_channel=64, all_dim=256):
        super(CCorrM, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False) #weight
        self.channel = all_channel
        self.dim = all_dim * all_dim
        self.conv1 = DSConv3x3(all_channel, all_channel, stride=1)
        self.conv2 = DSConv3x3(all_channel, all_channel, stride=1)

    def forward(self, exemplar, query):  # exemplar: f1, query: f2
        fea_size = query.size()[2:]
        exemplar = F.interpolate(exemplar, size=fea_size, mode="bilinear", align_corners=True)
        all_dim = fea_size[0] * fea_size[1]
        exemplar_flat = exemplar.view(-1, self.channel, all_dim)  # N,C1,H,W -> N,C1,H*W
        query_flat = query.view(-1, self.channel, all_dim)  # N,C2,H,W -> N,C2,H*W
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batchsize x dim x num, N,H*W,C1
        exemplar_corr = self.linear_e(exemplar_t)  # batchsize x dim x num, N,H*W,C1
        A = torch.bmm(query_flat, exemplar_corr)  # ChannelCorrelation: N,C2,H*W x N,H*W,C1 = N,C2,C1

        A1 = F.softmax(A.clone(), dim=2)  # N,C2,C1. dim=2 is row-wise norm. Sr
        B = F.softmax(torch.transpose(A, 1, 2), dim=2)  # N,C1,C2 column-wise norm. Sc
        query_att = torch.bmm(A1, exemplar_flat).contiguous()  # N,C2,C1 X N,C1,H*W = N,C2,H*W
        exemplar_att = torch.bmm(B, query_flat).contiguous()  # N,C1,C2 X N,C2,H*W = N,C1,H*W

        exemplar_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])  # N,C1,H*W -> N,C1,H,W
        exemplar_out = self.conv1(exemplar_att + exemplar)

        query_att = query_att.view(-1, self.channel, fea_size[0], fea_size[1])  # N,C2,H*W -> N,C2,H,W
        query_out = self.conv1(query_att + query)

        return exemplar_out, query_out


# Edge-based Enhancement Unit (EEU)
class EEU(nn.Module):
    def __init__(self, in_channel):
        super(EEU, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.PReLU = nn.PReLU(in_channel)

    def forward(self, x):
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight * x + x
        return self.PReLU(edge), out


class ESAM(nn.Module):
    def __init__(self, channel1=64, channel2=64):
        super(ESAM, self).__init__()

        self.smooth1 = DSConv3x3(channel1, channel2, stride=1, dilation=1)  # 16channel-> 24channel
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.smooth2 = DSConv3x3(channel2, channel2, stride=1, dilation=1)  # 24channel-> 24channel

        self.eeu1 = EEU(channel2)
        self.eeu2 = EEU(channel2)
        self.ChannelCorrelation = CCorrM(channel2, 128)

    def forward(self, x1, x2):  # x1 16*144*14; x2 24*72*72

        x1_1 = self.smooth1(x1)
        edge1, x1_2 = self.eeu1(x1_1)

        x2_1 = self.smooth2(self.upsample2(x2))
        edge2, x2_2 = self.eeu2(x2_1)

        # Channel-wise Correlation
        x1_out, x2_out = self.ChannelCorrelation(x1_2, x2_2)

        return edge1, edge2, torch.cat([x1_out, x2_out], 1)  # (24*2)*144*144

class CrossAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key1   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.query2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key2   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv_f

    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2)
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        q = torch.cat([q1,q2],1).view(batch_size, -1, height * width).permute(0, 2, 1)
        attn_matrix1 = torch.bmm(q, k1)
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1

        attn_matrix2 = torch.bmm(q, k2)
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2
        # feat_sum = self.conv_cat(torch.cat([out1,out2],1))
        return out1, out2
#Transormer Ecoder with x2, x4, x8, x16 scales
class EncoderTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=3, in_chans=3, num_classes=2, embed_dims=[32, 64, 128, 256],
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=LayerNorm, depths=[3, 3, 6, 18]):
        super().__init__()
        self.num_classes    = num_classes
        self.depths         = depths
        self.embed_dims     = embed_dims

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Stage-1 (x1/4 scale)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0 

        self.block1 = nn.ModuleList([EncoderBlock(dim=embed_dims[0], dim_head=4)
                                     for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        
        # Stage-2 (x1/8 scale)
        cur += depths[0]

        self.block2 = nn.ModuleList([EncoderBlock(dim=embed_dims[1], dim_head=4)
                                     for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
       
       # Stage-3 (x1/16 scale)
        cur += depths[1]
        
        self.block3 = nn.ModuleList([EncoderBlock(dim=embed_dims[2], dim_head=8)
                                     for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        
        # Stage-4 (x1/32 scale)
        cur += depths[2]

        self.block4 = nn.ModuleList([EncoderBlock(dim=embed_dims[3], dim_head=8)
                                     for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))

            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward_features(self, feats):
        # print(feats.shape) torch.Size([4, 3, 512, 512])

        B = feats.shape[0]
        outs = []
    
        # stage 1
        feats, H1, W1 = self.patch_embed1(feats)
        for i, blk in enumerate(self.block1):
            feats = blk(feats, H1, W1)
        feats = self.norm1(feats)
        # print(feats.shape) torch.Size([4, 64, 128, 128])
        #feats = feats.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feats)

        feats, H1, W1 = self.patch_embed2(feats)
        for i, blk in enumerate(self.block2):
            feats = blk(feats, H1, W1)
        feats = self.norm2(feats)
        # print(feats.shape) torch.Size([4, 128, 64, 64])
        #feats = feats.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feats)

        # stage 3
        feats, H1, W1 = self.patch_embed3(feats)
        for i, blk in enumerate(self.block3):
            feats = blk(feats, H1, W1)
        feats = self.norm3(feats)
        # print(feats.shape) torch.Size([4, 320, 32, 32])
        #feats = feats.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feats)

        # stage 4
        feats, H1, W1 = self.patch_embed4(feats)
        for i, blk in enumerate(self.block4):
            feats = blk(feats, H1, W1)
        feats = self.norm4(feats)
        # print(feats.shape) torch.Size([4, 512, 16, 16])
        #feats = feats.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feats)
        
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

class DecoderTransformer(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, align_corners=True, in_channels=[64, 128, 320, 512], embedding_dim=256, output_nc=2, decoder_softmax=False):
        super(DecoderTransformer, self).__init__()
        
        #settings
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.esam = ESAM(64, 128)
        self.ccorm = CCorrM(256, 256)
        self.croAtten1 = CrossAtt(64, 64)
        self.croAtten2 = CrossAtt(128, 128)
        self.croAtten3 = CrossAtt(320, 320)
        self.croAtten4 = CrossAtt(512, 512)
        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
           nn.Conv2d(in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )
      
        self.ceff1 = CEFF(in_channels=self.embedding_dim, height=2)
        self.ceff2 = CEFF(in_channels=self.embedding_dim, height=2)
        self.ceff3 = CEFF(in_channels=self.embedding_dim, height=2)
        self.ceff4 = CEFF(in_channels=self.embedding_dim, height=2)
        self.basiconv = BasicConv2d(512,256,1)
        self.basiconv1 = BasicConv2d(256,128,1)
        self.basiconvUP = BasicConv2d(128, 256, 1)



        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid()

    def forward(self, x_1, x_2):

        #img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1

        c1_2, c2_2, c3_2, c4_2 = x_2
        # print(c1_1.shape)  #64*128*128
        # print(c2_1.shape)  #128*64*64
        # input()
        # 对每一对特征进行crossattention
        c1_1, c1_2 = self.croAtten1(c1_1, c1_2)
        c2_1, c2_2 = self.croAtten2(c2_1, c2_2)
        c3_1, c3_2 = self.croAtten3(c3_1, c3_2)
        c4_1, c4_2 = self.croAtten4(c4_1, c4_2)
        # edge1, edge2, conv12 = self.esam(c1_2, c2_2)
        # edge1_2, edge2_2, conv12_2 = self.esam(c1_1, c2_1)
        # conv12 = self.ceff4([conv12, conv12_2])
        
        # print(conv12.shape) # 256 * 128*128
        # input()
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        # print(_c4_1.shape)#torch.Size([4, 256, 16, 16])


        _c4   = self.ceff1([_c4_1, _c4_2])
        #_c4 = self.basiconv(torch.cat([_c4_1, _c4_2], 1))
        p_c4  = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        # print(_c3_1.shape)#torch.Size([4, 256, 32, 32])

        _c3   = self.ceff2([_c3_1, _c3_2])
        # _c3 = self.basiconv(torch.cat([_c3_1, _c3_2], 1))
        p_c3  = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        # print(_c2_1.shape)#torch.Size([4, 256, 64, 64])

        _c2   = self.ceff3([_c2_1, _c2_2])
        # _c2 = self.basiconv(torch.cat([_c2_1, _c2_2], 1))
        p_c2  = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        # print(_c1_1.shape)#torch.Size([4, 256, 128, 128])

        # _c1 = self.basiconv(torch.cat([_c1_1, _c1_2], 1))
        _c1 = self.ceff4([_c1_1, _c1_2])
        # 方法一：边缘特征跟卷积降维后的stage1特征cat起来，作为stage1的输出

        # _c1 = torch.cat([_c1, conv12], 1)
        # _c1 = self.basiconv(_c1)
        # print(_c1.shape)#torch.Size([4, 256, 128, 128])

        p_c1  = self.make_pred_c1(_c1) # torch.Size([4, 2, 128, 128])
        outputs.append(p_c1)

        #Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat([_c4_up, _c3_up, _c2_up, _c1], dim=1)) # torch.Size([4, 256, 128, 128])
        # print(_c.shape)
        # print(conv12.shape)
        # input()
        # _c = torch.cat([_c, conv12], 1)
        # _c = self.basiconv(_c)
      
        #Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        #Residual block
        x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        # print(cp.shape)# torch.Size([4, 2, 512, 512])
        # input()
        # outputs.append(edge1)
        # outputs.append(edge2)
        # outputs.append(edge1_2)
        # outputs.append(edge2_2)
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


# ScratchFormer:
class ScratchFormer(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256):
        super(ScratchFormer, self).__init__()
        # Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths = [2, 2, 2, 2]
        self.embedding_dim = embed_dim
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1 

        self.Tenc_x2 = EncoderTransformer(patch_size = 7, in_chans=input_nc, num_classes=output_nc, embed_dims=self.embed_dims,
                                             attn_drop_rate = self.attn_drop, drop_path_rate=self.drop_path_rate,
                                             norm_layer=partial(LayerNorm, eps=1e-6), depths=self.depths)
        # self.Tenc_x2 = EncoderTransformer(patch_size=7, in_chans=input_nc, num_classes=output_nc,
        #                                embed_dims=self.embed_dims,
        #                                attn_drop_rate=self.attn_drop, drop_path_rate=self.drop_path_rate,
        #                                norm_layer=partial(LayerNorm, eps=1e-6), depths=self.depths)

        #Transformer Decoder
        self.TDec_x2 = DecoderTransformer(align_corners=False, in_channels = self.embed_dims, embedding_dim= self.embedding_dim,
                                            output_nc=output_nc, decoder_softmax = decoder_softmax)
        # self.TDec_x2 = DecoderTransformer(align_corners=False, in_channels=self.embed_dims,
        #                                embedding_dim=self.embedding_dim,
        #                                output_nc=output_nc, decoder_softmax=decoder_softmax)

    def forward(self, x1, x2):

        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]
        # fx1 = self.tenc(x1)
        # fx2 = self.tenc(x2)
        # cp = self.tdec(fx1, fx2)
        cp = self.TDec_x2(fx1,fx2)
        return cp
