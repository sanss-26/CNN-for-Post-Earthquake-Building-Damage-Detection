import torch
import numpy as np
import h5py
import os
import cv2
import kornia


def clean_shadow_mask(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    Clean a binary shadow mask using morphological operations and connected component filtering.

    Args:
        mask: HxW binary numpy array with values 0 or 255.
        min_area: minimum area (in pixels) for a region to be kept.

    Returns:
        clean: HxW binary numpy array with cleaned mask values 0 or 255.
    """
    mask_bin = (mask > 127).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    clean = np.zeros_like(mask_bin)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == label] = 255
    return clean


def extract_shadow_mask(opt_image: np.ndarray) -> np.ndarray:
    # Convert to HxWx3 uint8
    img = (np.transpose(opt_image, (1, 2, 0)) * 255).astype(np.uint8)
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # ── Exp 3 CLAHE settings ──────────────────────
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)
    # Otsu threshold
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ── Morphological cleaning (kernel=5, 1 open, 1 close) ──
    mask_bin = (mask > 127).astype(np.uint8) * 255
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opened   = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN,  kernel,
                                iterations=1)
    closed   = cv2.morphologyEx(opened,   cv2.MORPH_CLOSE, kernel,
                                iterations=1)
    # Connected-component filtering (min_area=100)
    clean = clean_shadow_mask(closed, min_area=100)
    # Normalize & channel
    shadow = (clean.astype(np.float32) / 255.0)[None, ...]
    return shadow


class EarthquakeDataset(torch.utils.data.Dataset):
    '''
    Dataset class for earthquake building damage classification.
    '''
    def __init__(self, root, splits=['fold-2.txt','fold-3.txt','fold-4.txt','fold-5.txt']):
        self.root = root
        self.splits = splits
        self.ids = []
        self.labels = []
        for split in splits:
            split_file = os.path.join(root, split)
            with open(split_file, 'r') as f:
                for line in f:
                    fid, lbl = line.strip().split(',')
                    self.ids.append(fid)
                    self.labels.append(int(lbl))
        self.length = len(self.ids)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        osmid = self.ids[index]
        label = np.float32(self.labels[index])

        # Load SAR
        sar_path = os.path.join(self.root, osmid + '_SAR.mat')
        with h5py.File(sar_path, 'r') as f1:
            x1 = np.float32(f1['x1'])
        x1[x1 < -100] = x1[x1 > -100].min()
        x1 = cv2.resize(x1, (224, 224), interpolation=cv2.INTER_LINEAR_EXACT)
        p1, p99 = np.percentile(x1, (1, 99))
        x1 = np.clip(x1, p1, p99)
        x1 = (x1 - p1) / (p99 - p1)
        x1 = np.stack((x1,)*3, axis=0)

        # Load SARftp
        sarftp_path = os.path.join(self.root, osmid + '_SARftp.mat')
        with h5py.File(sarftp_path, 'r') as f2:
            x2 = np.float32(f2['x2'])
        x2 = cv2.resize(x2, (224, 224), interpolation=cv2.INTER_NEAREST)
        x2 = np.stack((x2,)*3, axis=0)

        # Load optical
        opt_path = os.path.join(self.root, osmid + '_opt.mat')
        with h5py.File(opt_path, 'r') as f3:
            x3 = np.float32(f3['x3'])
        x3 = cv2.resize(np.transpose(x3, (1,2,0)), (224,224), interpolation=cv2.INTER_LINEAR_EXACT)
        x3 = np.transpose(x3, (2,0,1))
        p1, p99 = np.percentile(x3, (1, 99))
        x3 = np.clip(x3, p1, p99)
        x3 = (x3 - p1) / (p99 - p1)

        # Load optical ftp
        optftp_path = os.path.join(self.root, osmid + '_optftp.mat')
        with h5py.File(optftp_path, 'r') as f4:
            x4 = np.float32(f4['x4'])
        x4 = cv2.resize(x4, (224, 224), interpolation=cv2.INTER_NEAREST)
        x4 = np.stack((x4,)*3, axis=0)

        # Generate shadow mask with CLAHE + cleaning
        shadow_mask = extract_shadow_mask(x3)
        opt_with_shadow = np.concatenate((x3, shadow_mask), axis=0)

        # Convert all to float tensors
        images = {
            'sar': torch.from_numpy(x1).float(),
            'sarftp': torch.from_numpy(x2).float(),
            'opt': torch.from_numpy(x3).float(),
            'opt_with_shadow': torch.from_numpy(opt_with_shadow).float(),
            'optftp': torch.from_numpy(x4).float()
        }

        return images, torch.tensor(label).float()

class MyAug(torch.nn.Module):
    """
    A custom augmentation module using Kornia.
    """
    def __init__(self):
        super(MyAug, self).__init__()
        self.rand_crop = kornia.augmentation.RandomResizedCrop((224, 224), scale=(0.8, 1.0))
        self.hflip   = kornia.augmentation.RandomHorizontalFlip(p=0.5)
        self.vflip   = kornia.augmentation.RandomVerticalFlip(p=0.5)

    def forward(self, sar, sarftp, opt, optftp):
        sar    = self.rand_crop(sar)
        sarftp = self.rand_crop(sarftp)
        opt    = self.rand_crop(opt)
        optftp = self.rand_crop(optftp)

        sar    = self.hflip(sar)
        sarftp = self.hflip(sarftp)
        opt    = self.hflip(opt)
        optftp = self.hflip(optftp)

        sar    = self.vflip(sar)
        sarftp = self.vflip(sarftp)
        opt    = self.vflip(opt)
        optftp = self.vflip(optftp)

        return sar, sarftp, opt, optftp