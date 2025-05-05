import torch
import torch.nn as nn
import torchvision.models as models


class MODEL_OPT(nn.Module):
    """
    A late fusion model that combines optical image and optical FTP features.
    If use_shadow is True, the optical branch accepts a 4-channel input (RGB + shadow).
    """

    def __init__(self, pretrain=None, use_shadow=False):
        super(MODEL_OPT, self).__init__()
        self.use_shadow = use_shadow

        # Determine whether to use pretrained weights
        self.pretrained = pretrain is not None

        # Initialize the optical branch with ResNet18
        self.model_opt = models.resnet18(pretrained=self.pretrained)
        # Modify first conv layer if using 4-channel input
        if self.use_shadow:
            old_conv = self.model_opt.conv1
            self.model_opt.conv1 = nn.Conv2d(
                4,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            with torch.no_grad():
                self.model_opt.conv1.weight[:, :3, :, :] = old_conv.weight
                self.model_opt.conv1.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

        self.model_opt.fc = nn.Linear(512, 128)

        self.model_ftp = models.resnet18(pretrained=self.pretrained)
        self.model_ftp.fc = nn.Linear(512, 128)

        # Final classification layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(),
            nn.Linear(128, 1)
        )

    def forward(self, opt, ftp, return_features=False):
        x_opt = self.model_opt(opt)
        x_ftp = self.model_ftp(ftp)
        features = torch.cat((x_opt, x_ftp), dim=1)
        if return_features:
            return features
        x = self.fc(features)
        return x


class MODEL_SAR(nn.Module):
    '''
    Late fusion model class for SAR and SARftp.
    '''

    def __init__(self, pretrain=None):
        super(MODEL_SAR, self).__init__()
        self.model_sar = models.resnet18()
        self.model_sar.fc = nn.Linear(512, 128)

        self.model_ftp = models.resnet18(pretrained=True)
        self.model_ftp.fc = nn.Linear(512, 128)

        if pretrain is not None:
            ckpt = torch.load(pretrain)
            ckpt.pop('fc.weight', None)
            ckpt.pop('fc.bias', None)
            msg = self.model_sar.load_state_dict(ckpt, strict=False)
            print(msg)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(),
            nn.Linear(128, 1)
        )

    def forward(self, sar, ftp):
        x_sar = self.model_sar(sar)
        x_ftp = self.model_ftp(ftp)
        x = torch.cat((x_sar, x_ftp), dim=1)
        x = self.fc(x)
        return x


class MODEL_MM(nn.Module):
    """
    A late fusion model that combines SAR, SARftp, and optical branches.
    The optical branch uses MODEL_OPT which accepts a 4-channel input if use_shadow is True.
    """

    def __init__(self, sar_pretrain=None, opt_pretrain=None, use_shadow=False):
        super(MODEL_MM, self).__init__()

        # SAR branch
        self.model_sar = models.resnet18(pretrained=False)
        if sar_pretrain is not None:
            ckpt = torch.load(sar_pretrain, map_location=torch.device('cpu'))
            ckpt.pop('fc.weight', None)
            ckpt.pop('fc.bias', None)
            self.model_sar.load_state_dict(ckpt, strict=False)
        self.model_sar.fc = nn.Linear(512, 128)

        # SARFTP branch
        self.model_sarftp = models.resnet18(pretrained=True)
        self.model_sarftp.fc = nn.Linear(512, 128)

        # Optical branch using custom MODEL_OPT
        self.model_opt = MODEL_OPT(opt_pretrain, use_shadow=use_shadow)

        # Final fusion fully-connected layers:
        # x_sar (128) + x_sarftp (128) + x_opt+ftp (256) = 512
        self.fc = nn.Sequential(
            nn.Linear(128 * 4, 128),  # 128*4 = 512
            nn.Dropout(),
            nn.Linear(128, 1)
        )


    def forward(self, sar, sarftp, opt, optftp):
        # print("[Model input] opt shape:", opt.shape)

        x_sar = self.model_sar(sar)
        x_sarftp = self.model_sarftp(sarftp)
        x_opt = self.model_opt(opt, optftp, return_features=True)  # returns 256-dim feature
        x = torch.cat((x_sar, x_sarftp, x_opt), dim=1)  # final shape: [batch, 512]
        x = self.fc(x)
        return x
