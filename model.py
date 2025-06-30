import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CrossModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_conv = nn.Conv2d(dim, dim//8, 1)
        self.key_conv   = nn.Conv2d(dim, dim//8, 1)
        self.value_conv = nn.Conv2d(dim, dim,     1)
        self.gamma      = nn.Parameter(torch.zeros(1))

    def forward(self, x_opt, x_sar):
        b,c,h,w = x_opt.shape
        q = self.query_conv(x_opt).view(b,-1,h*w)           # B×C'×N
        k = self.key_conv(x_sar).view(b,-1,h*w)             # B×C'×N
        v = self.value_conv(x_sar).view(b,-1,h*w)           # B×C ×N
        attn = torch.softmax(torch.bmm(q.permute(0,2,1), k), dim=-1)  # B×N×N
        out  = torch.bmm(v, attn.permute(0,2,1)).view(b,c,h,w)
        return self.gamma*out + x_opt

class M3ICNet(nn.Module):
    def __init__(self, sar_pretrain=None, opt_pretrain=None, use_shadow=False):
        super().__init__()
        self.use_shadow = use_shadow

        # optical (RGB) encoder
        opt_resnet = models.resnet18(pretrained=(opt_pretrain is not None))
        self.opt_encoder = nn.Sequential(*list(opt_resnet.children())[:-2])  # B×512×7×7

        # SAR encoder
        opt_resnet = models.resnet50(pretrained=(opt_pretrain is not None))
        sar_resnet = models.resnet50(pretrained=(sar_pretrain is not None))

        # then strip off its final layers just like before:
        self.opt_encoder = nn.Sequential(*list(opt_resnet.children())[:-2])  # B×2048×7×7
        self.sar_encoder = nn.Sequential(*list(sar_resnet.children())[:-2])  # B×2048×7×7

        # fusion block still takes dim=2048 now:
        self.fusion = CrossModalFusion(dim=2048)

        # shadow branch output must match 2048 channels:
        if use_shadow:
            self.shadow_branch = nn.Sequential(
                nn.Conv2d(1, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 2048, 1),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((7,7))
            )

        # classifier now pools 2048→1:
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, sar, sarftp, opt, optftp=None):
        # split RGB and shadow channels
        if self.use_shadow:
            rgb    = opt[:,:3]
            shadow = opt[:,3:4]
        else:
            rgb, shadow = opt, None

        fo = self.opt_encoder(rgb)   # B×512×7×7
        fs = self.sar_encoder(sar)   # B×512×7×7

        fused = self.fusion(fo, fs)

        # add shadow features
        if shadow is not None:
            sf = self.shadow_branch(shadow)  # B×512×7×7
            fused = fused + sf

        x = self.pool(fused)  
        logits = self.classifier(x)
        return logits

# Legacy multimodal classes (unchanged)
class MODEL_OPT(nn.Module):
    def __init__(self, pretrain=None, use_shadow=False):
        super().__init__()
        self.use_shadow = use_shadow
        base = models.resnet18(pretrained=(pretrain is not None))
        if use_shadow:
            old = base.conv1
            base.conv1 = nn.Conv2d(4, old.out_channels,
                                   kernel_size=old.kernel_size,
                                   stride=old.stride,
                                   padding=old.padding,
                                   bias=old.bias is not None)
            with torch.no_grad():
                base.conv1.weight[:, :3] = old.weight
                base.conv1.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
        self.model_opt = base
        self.model_opt.fc = nn.Linear(512, 128)
        ftp = models.resnet18(pretrained=(pretrain is not None))
        ftp.fc = nn.Linear(512, 128)
        self.model_ftp = ftp
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.Dropout(), nn.Linear(128, 1))

    def forward(self, opt, ftp, return_features=False):
        x1 = self.model_opt(opt)
        x2 = self.model_ftp(ftp)
        feat = torch.cat([x1, x2], dim=1)
        return feat if return_features else self.fc(feat)

class MODEL_SAR(nn.Module):
    def __init__(self, pretrain=None):
        super().__init__()
        sar = models.resnet18(pretrained=False)
        sar.fc = nn.Linear(512, 128)
        ftp = models.resnet18(pretrained=True)
        ftp.fc = nn.Linear(512, 128)
        self.model_sar = sar
        self.model_ftp = ftp
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.Dropout(), nn.Linear(128, 1))
        if pretrain:
            ck = torch.load(pretrain, map_location='cpu')
            ck.pop('fc.weight', None);
            ck.pop('fc.bias', None)
            self.model_sar.load_state_dict(ck, strict=False)

    def forward(self, sar, ftp):
        x1 = self.model_sar(sar)
        x2 = self.model_ftp(ftp)
        return self.fc(torch.cat([x1, x2], dim=1))

class MODEL_MM(nn.Module):
    def __init__(self, sar_pretrain=None, opt_pretrain=None, use_shadow=False):
        super().__init__()
        sar = models.resnet18(pretrained=False)
        sar.fc = nn.Linear(512, 128)
        if sar_pretrain:
            ck = torch.load(sar_pretrain, map_location='cpu')
            ck.pop('fc.weight', None);
            ck.pop('fc.bias', None)
            sar.load_state_dict(ck, strict=False)
        sarftp = models.resnet18(pretrained=True)
        sarftp.fc = nn.Linear(512, 128)
        self.model_sar    = sar
        self.model_sarftp = sarftp
        self.model_opt    = MODEL_OPT(opt_pretrain, use_shadow)
        self.fc = nn.Sequential(nn.Linear(512, 128), nn.Dropout(), nn.Linear(128, 1))

    def forward(self, sar, sarftp, opt, optftp):
        x1 = self.model_sar(sar)
        x2 = self.model_sarftp(sarftp)
        x3 = self.model_opt(opt, optftp, return_features=True)
        return self.fc(torch.cat([x1, x2, x3], dim=1))
