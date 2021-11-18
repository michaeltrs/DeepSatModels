import torch
import torch.nn as nn
from utils.config_files_utils import get_params_values
from models.LocalSelfAttention.cscl import ContextSelfSimilarity, AttentionAggregate


def conv_block(in_dim, middle_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, middle_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(middle_dim),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


def center_in(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True))
    return model


def center_out(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(in_dim),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1))
    return model


def up_conv_block_3d(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


def up_conv_block_2d(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


def conv_block_2d(in_dim, middle_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, middle_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(middle_dim),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(middle_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


class UNet3D_CSCL(nn.Module):
    """
    UNet3D Self Attention Loss Head
    """
    
    def __init__(self, config):
        # in_channel, n_classes, timesteps, dropout
        super(UNet3D_CSCL, self).__init__()
        self.attn_channels = get_params_values(config, "attn_channels", 128)  # config['attn_channels']
        self.cscl_win_size = get_params_values(config, "cscl_win_size", 3)  # config['cscl_win_size']
        self.cscl_win_dilation = get_params_values(config, "cscl_win_dilation", 1)  # config['cscl_win_dilation']
        self.cscl_win_stride = get_params_values(config, "cscl_win_stride", 1)  # config['cscl_win_stride']
        self.groups = get_params_values(config, "attn_groups", 1)  # config['attn_groups']
        self.timesteps = get_params_values(config, "max_seq_len")
        self.shape_pattern = get_params_values(config, "shape_pattern", "NCTHW")
        self.num_classes = config['num_classes']
        self.stage = config['train_stage']
        self.backbone_arch = get_params_values(config, 'backbone', 'UNET3D')
        self.norm_emb = get_params_values(config, 'norm_emb', False)
        self.backbone = UNet3Df(config)
        self.emb_channels = get_params_values(config, "emb_channels", self.attn_channels)  # self.backbone.final.out_channels
        self.output_magnification = get_params_values(config, "output_magnification", None)

        # Build model --------------------------------------------------------------------------------------------------
        if self.output_magnification in [2, 4]:
            self.conv_out_x2 = up_conv_block_2d(self.emb_channels, self.emb_channels)
        if self.output_magnification in [4]:
            self.conv_out_x4 = up_conv_block_2d(self.emb_channels, self.emb_channels)

        if self.stage in [0, 3, 4]:
            self.attn_sim = ContextSelfSimilarity(
                in_channels=self.emb_channels, attn_channels=self.attn_channels, kernel_size=self.cscl_win_size,
                stride=self.cscl_win_stride, dilation=self.cscl_win_dilation, groups=self.groups, bias=False,
                norm_emb=self.norm_emb)
            if self.stage == 3:
                self.linear_out = nn.Conv2d(in_channels=self.emb_channels + self.cscl_win_size ** 2,
                                            out_channels=self.num_classes, kernel_size=1, stride=1, padding=0)
            if self.stage == 4:
                self.op_out = get_params_values(config, 'op_out', "sum")
                self.attn_agg = AttentionAggregate(
                    in_channels=self.emb_channels, out_channels=self.emb_channels, attn_channels=self.attn_channels,
                    kernel_size=self.cscl_win_size, stride=self.cscl_win_stride, dilation=self.cscl_win_dilation,
                    groups=1, bias=False, norm_emb=self.norm_emb, out_op=self.op_out)
                self.linear_out = nn.Conv2d(in_channels=self.emb_channels, out_channels=self.num_classes, kernel_size=1,
                                            stride=1, padding=0)

        if self.stage == 2:
            self.linear_out = nn.Linear(self.emb_channels, self.num_classes)

        self.trainable_params = list(self.parameters())

    def forward(self, x, out_val=None):
        emb = self.backbone(x)

        if self.output_magnification in [2, 4]:
            emb = self.conv_out_x2(emb)
        if self.output_magnification in [4]:
            emb = self.conv_out_x4(emb)

        #print("embedding: ", emb.shape)
        if self.stage == 0:
            return self.attn_sim(emb)
        # if out_val == "attn_emb":
        #     return self.attn_sim.local_agg(emb) #+ emb
        if self.stage == 2:
            logits = self.linear_out(emb.permute(0, 2, 3, 1))#.reshape(-1, self.num_classes)
            logits = logits.permute(0, 3, 1, 2)
            return logits
        elif self.stage in [3, 4]:
            sim = self.attn_sim(emb)
            if self.stage == 3:
                emb_ = emb[:, :, ::self.stride, ::self.stride]
                bs, c, h, w = emb_.shape
                sim = sim.reshape(bs, h, w, self.kernel_size**2).permute(0, 3, 1, 2)
                emb = torch.cat([emb_, sim], dim=1)
            else:
                emb = self.attn_agg(emb, sim)
            # print("emb: ", emb.shape)
            return self.linear_out(emb)


class UNet3Df(nn.Module):
    def __init__(self, config):
        super(UNet3Df, self).__init__()
        in_channel = get_params_values(config, "num_channels")
        self.timesteps = get_params_values(config, "max_seq_len")
        self.shape_pattern = get_params_values(config, "shape_pattern", "NCTHW")
        
        feats = get_params_values(config, "num_features", 16)
        self.en3 = conv_block(in_channel, feats * 4, feats * 4)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.en4 = conv_block(feats * 4, feats * 8, feats * 8)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.center_in = center_in(feats * 8, feats * 16)
        self.center_out = center_out(feats * 16, feats * 8)
        self.dc4 = conv_block(feats * 16, feats * 8, feats * 8)
        self.trans3 = up_conv_block_3d(feats * 8, feats * 4)

        self.dc3 = conv_block(feats * 8, feats * 8, feats * 8)                            # different than Unet3D
        self.final = nn.Conv3d(feats * 8, feats * 8, kernel_size=3, stride=1, padding=1)  # different than Unet3D

        self.fn = nn.Linear(self.timesteps, 1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if self.shape_pattern == "NTHWC":
            x = x.permute(0, 4, 1, 2, 3)
        assert x.shape[2] == self.timesteps, "Input to UNET3D temporal dimension should equal %d, here %d" \
                                             % (self.timesteps, x.shape[2])
        # x = x.cuda()
        en3 = self.en3(x)
        # print("en3: ", en3.shape)
        pool_3 = self.pool_3(en3)
        # print("pool_3: ", pool_3.shape)
        en4 = self.en4(pool_3)
        # print("en4: ", en4.shape)
        pool_4 = self.pool_4(en4)
        # print("pool_4: ", pool_4.shape)
        center_in = self.center_in(pool_4)
        # print("center_in ", center_in.shape)
        center_out = self.center_out(center_in)
        # print("center_out: ", center_out.shape)
        concat4 = torch.cat([center_out, en4], dim=1)
        # print("concat4: ", concat4.shape)
        dc4 = self.dc4(concat4)
        # print("dc4: ", dc4.shape)
        trans3 = self.trans3(dc4)
        # print("trans3: ", trans3.shape)
        concat3 = torch.cat([trans3, en3], dim=1)
        # print("concat3: ", concat3.shape)
        dc3 = self.dc3(concat3)
        # print("dc3: ", dc3.shape)
        final = self.final(dc3)
        # print("final: ", final.shape)
        final = final.permute(0, 1, 3, 4, 2)
        # print("final: ", final.shape)
        
        shape_num = final.shape[0:4]
        final = final.reshape(-1, final.shape[4])
        # print("final: ", final.shape)
        final = self.fn(final)
        # print("final: ", final.shape)
        final = final.reshape(shape_num)
        # print("final: ", final.shape)
        return final


class UNet3Dsmall_backbone(nn.Module):
    def __init__(self, config):
        # in_channel, n_classes, timesteps, dropout
        super(UNet3Dsmall_backbone, self).__init__()
        in_channel = get_params_values(config, "num_channels")
        self.timesteps = get_params_values(config, "max_seq_len")
        self.shape_pattern = get_params_values(config, "shape_pattern", "NCTHW")

        feats = get_params_values(config, "num_features", 16)
        self.en3 = conv_block(in_channel, feats * 4, feats * 4)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.en4 = conv_block(feats * 4, feats * 8, feats * 8)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.center_in = center_in(feats * 8, feats * 16)
        self.center_out = center_out(feats * 16, feats * 8)
        self.dc4 = conv_block(feats * 16, feats * 8, feats * 8)
        self.trans3 = up_conv_block_3d(feats * 8, feats * 4)

        self.dc3 = conv_block(feats * 8, feats * 4, feats * 2)

        self.fn = nn.Linear(self.timesteps, 1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if self.shape_pattern == "NTHWC":
            x = x.permute(0, 4, 1, 2, 3)
        assert x.shape[2] == self.timesteps, "Input to UNET3D temporal dimension should equal %d, here %d" \
                                             % (self.timesteps, x.shape[2])
        # x = x.cuda()
        en3 = self.en3(x)
        # print("en3: ", en3.shape)
        pool_3 = self.pool_3(en3)
        # print("pool_3: ", pool_3.shape)
        en4 = self.en4(pool_3)
        # print("en4: ", en4.shape)
        pool_4 = self.pool_4(en4)
        # print("pool_4: ", pool_4.shape)
        center_in = self.center_in(pool_4)
        # print("center_in ", center_in.shape)
        center_out = self.center_out(center_in)
        # print("center_out: ", center_out.shape)
        concat4 = torch.cat([center_out, en4], dim=1)
        # print("concat4: ", concat4.shape)
        dc4 = self.dc4(concat4)
        # print("dc4: ", dc4.shape)
        trans3 = self.trans3(dc4)
        # print("trans3: ", trans3.shape)
        concat3 = torch.cat([trans3, en3], dim=1)
        # print("concat3: ", concat3.shape)
        dc3 = self.dc3(concat3)
        # print("dc3: ", dc3.shape)
        # final = self.final(dc3)
        # print("final: ", final.shape)
        final = dc3.permute(0, 1, 3, 4, 2)
        # print("final: ", final.shape)

        shape_num = final.shape[0:4]
        final = final.reshape(-1, final.shape[4])
        # print("final: ", final.shape)
        final = self.fn(final)
        # print("final: ", final.shape)
        final = final.reshape(shape_num)
        # print("final: ", final.shape)
        return final
