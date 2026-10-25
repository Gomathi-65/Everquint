"""Two CNNs trained from scratch. They differ in ONE thing: the residual (skip) connection."""
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class ResidualBlock(nn.Module):
    """Simplified ResNet basic block: out = ReLU( BN(conv(ReLU(BN(conv(x))))) + shortcut(x) ).

    It has exactly the two 3x3 convolutions of one baseline stage, plus the skip connection.
    When the channel count changes the shortcut is a 1x1 convolution + BatchNorm.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CNN(nn.Module):
    """3 stages (32, 64, 128 channels) -> global average pool -> Dropout -> Linear classifier.

    stage = [2 x (Conv3x3-BN-ReLU)] (baseline)  or  [ResidualBlock] (residual), then MaxPool and Dropout2d.
    """

    def __init__(self, residual=False, num_classes=10, widths=(32, 64, 128), dropout=0.3, conv_dropout=0.1):
        super().__init__()
        layers, in_ch = [], 3
        for w in widths:
            layers += [ResidualBlock(in_ch, w)] if residual else [conv_bn_relu(in_ch, w), conv_bn_relu(w, w)]
            layers += [nn.MaxPool2d(2), nn.Dropout2d(conv_dropout)]
            in_ch = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, **kw):
    if name not in ("baseline", "residual"):
        raise ValueError(f"unknown model {name!r}")
    return CNN(residual=(name == "residual"), **kw)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
