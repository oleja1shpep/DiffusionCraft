import torch
from torch import nn


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, width, height, length = q.shape
        q = q.reshape(b, c, -1)
        q = q.permute(0, 2, 1)  # b,whl,c
        k = k.reshape(b, c, -1)  # b,c,whl
        w_ = torch.bmm(q, k)  # b,whl,whl    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, -1)
        w_ = w_.permute(0, 2, 1)  # b,whl,whl (first whl of k, second of q)
        h_ = torch.bmm(
            v, w_
        )  # b, c,whl (whl of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, width, height, length)

        h_ = self.proj_out(h_)

        return x + h_


class ResnetBlock3D(nn.Module):
    def __init__(self, *, in_channels, out_channels, conv_shortcut=False, dropout=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x: torch.Tensor):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h
