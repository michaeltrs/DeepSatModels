# modified from: https://github.com/leaderj1001/Stand-Alone-Self-Attention/blob/master/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ContextSelfSimilarity(nn.Module):
    def __init__(self, in_channels, attn_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 norm_emb=False, sigmoid_sim=False):
        super(ContextSelfSimilarity, self).__init__()
        self.attn_channels = attn_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # assert self.final_stride >= 1.0, "CSSL stride should be geq to dilation rate for special case to work"
        if self.stride >= self.dilation:
            assert self.stride % self.dilation == 0, \
                "CSCL stride should be integer multiple of dilation rate for special case to work"
            self.first_stride = self.dilation
            self.final_stride = int(self.stride / self.dilation)
            self.final_dilation = 1
            self.final_kernel_size = self.kernel_size
        elif self.stride < self.dilation:
            assert self.dilation % self.stride == 0, \
                "CSCL dilation should be integer multiple of stride for special case to work"
            self.first_stride = self.stride
            self.final_stride = 1
            self.final_dilation = int(self.dilation / self.stride)
            self.final_kernel_size = (self.kernel_size - 1) * self.final_dilation + 1

        self.padding = self.final_kernel_size // 2

        self.groups = groups
        self.norm_emb = norm_emb
        self.sigmoid_sim = sigmoid_sim

        assert self.attn_channels % self.groups == 0, "attn_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(attn_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(attn_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, attn_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, attn_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = x[:, :, ::self.first_stride, ::self.first_stride]

        batch, channels, height, width = x.shape

        q_out = self.query_conv(x[:, :, ::self.final_stride, ::self.final_stride])
        q_out = q_out.view(batch, self.groups, self.attn_channels // self.groups, height, width, 1)
        q_out = q_out.permute(0, 1, 3, 4, 5, 2)

        k_out = self.key_conv(x)
        k_out = F.pad(k_out, [self.padding, self.padding, self.padding, self.padding], value=0)
        k_out = self.unfold2D(k_out)
        k_out = k_out[:, :, :, :, ::self.final_dilation, ::self.final_dilation]

        k_out_h, k_out_w = k_out.split(self.attn_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        k_out = k_out.contiguous().view(batch, self.groups, self.attn_channels // self.groups, height, width, -1)
        k_out = k_out.permute(0, 1, 3, 4, 2, 5)

        if self.norm_emb:
            q_out = F.normalize(q_out, p=2, dim=5)
            k_out = F.normalize(k_out, p=2, dim=4)
        height1, width1 = q_out.shape[2:4]

        sim = torch.matmul(q_out, k_out)
        sim = sim.sum(dim=1).reshape(batch, height1, width1, self.kernel_size, self.kernel_size)
        if self.sigmoid_sim:
            sim = F.sigmoid(sim)

        return sim

    def unfold2D(self, x):
        return x.unfold(2, size=self.final_kernel_size, step=self.final_stride)\
                .unfold(3, size=self.final_kernel_size, step=self.final_stride)

    def local_agg(self, x):
        batch, channels, height, width = x.size()
        x_win = torch.nn.functional.pad(x, [self.padding, self.padding, self.padding, self.padding]). \
            unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        height1, width1 = x_win.shape[-4:-2]
        x_win = x_win.reshape(batch, channels, height1, width1, self.kernel_size ** 2).permute(0, 2, 3, 4, 1)
        sim = self.forward(x).reshape(batch, height1, width1, 1, self.kernel_size ** 2)

        out = torch.matmul(torch.softmax(sim, dim=-1), x_win)
        out = out.squeeze(3).permute(0, 3, 1, 2)
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class AttentionAggregate(ContextSelfSimilarity):
    def __init__(self, in_channels, out_channels, attn_channels, kernel_size, stride=1, dilation=1, groups=1,
                 bias=False, norm_emb=False, out_op="sum"):
        super(AttentionAggregate, self).__init__(in_channels, attn_channels, kernel_size, stride, dilation, groups,
                                                 bias, norm_emb)
        self.out_channels = out_channels
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        torch.nn.init.zeros_(self.value_conv.weight)

        if out_op == "sum":
            self.out_op = lambda x, y: x + y
        elif out_op == "cat":
            self.out_op = lambda x, y: torch.cat((x, y), dim=1)

    def forward(self, x, s):
        b, h, w, hs, ws = s.shape
        s = F.softmax(s.reshape(b, h, w, -1), dim=-1).unsqueeze(-1)
        x = x[:, :, ::self.dilation, ::self.dilation]
        v_out = self.value_conv(x)
        v_out = F.pad(v_out, [self.padding, self.padding, self.padding, self.padding], value=0)
        v_out = self.unfold2D(v_out)
        v_out = v_out[:, :, ::self.final_stride, ::self.final_stride, :, :]
        v_out = v_out.reshape(b, self.out_channels, h, w, -1).permute(0, 2, 3, 1, 4)
        out = torch.matmul(v_out, s).squeeze(-1).permute(0, 3, 1, 2)
        return self.out_op(out, x)


# ----------------------------------------------------------------------------------------------------------------------#

class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.dilated_kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.dilated_kernel_size // 2
        self.center_idx = kernel_size // 2

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = self.unfold2D(k_out)
        v_out = self.unfold2D(v_out)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def unfold2D(self, x):
        return x.unfold(2, self.dilated_kernel_size, self.stride) \
                   .unfold(3, self.dilated_kernel_size, self.stride)[:, :, :, :, ::self.dilation, ::self.dilation]

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)
