import torch
import numpy as np
import h5py
import os
import cv2
import kornia\

    

def extract_shadow_mask(opt_image):
    """
    Convert an optical image to grayscale and generate a binary shadow mask using Otsu's thresholding.
    Input:
      - opt_image: numpy array of shape (3, H, W) in range [0, 1] (channels-first)
    Output:
      - shadow_mask: numpy array of shape (1, H, W) with values 0 or 1
    """
    # Convert from (3, H, W) to (H, W, 3)
    opt_image_t = np.transpose(opt_image, (1, 2, 0))
    # If the image is in float format [0,1], convert it to uint8
    opt_image_uint8 = (opt_image_t * 255).astype(np.uint8)
    # Convert to grayscale
    gray_image = cv2.cvtColor(opt_image_uint8, cv2.COLOR_RGB2GRAY)
    # Apply Otsu's thresholding to get a binary mask
    _, shadow_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Normalize the mask to [0, 1] and add a channel dimension to get shape (1, H, W)
    shadow_mask = shadow_mask.astype(np.float32) / 255.0
    shadow_mask = np.expand_dims(shadow_mask, axis=0)
    return shadow_mask

class EarthquakeDataset(torch.utils.data.Dataset):
    '''
    Dataset class for earthquake building dataset.
    '''
    def __init__(self, root, splits=['fold-2.txt','fold-3.txt','fold-4.txt','fold-5.txt']):
        self.root = root
        self.splits = splits
        
        self.ids = []
        self.labels = []
        for split in splits:
            with open(os.path.join(root, split), 'r') as f:
                for line in f.readlines():
                    fid = line.split(',')[0]
                    label = line.split(',')[1]
                    self.ids.append(fid)
                    self.labels.append(int(label))
                
        self.length = len(self.ids)
        
    def __getitem__(self, index):
        osmid = self.ids[index]
        label = np.float32(self.labels[index])
        
        sar_path = os.path.join(self.root, osmid + '_SAR.mat')
        sarftp_path = os.path.join(self.root, osmid + '_SARftp.mat')
        opt_path = os.path.join(self.root, osmid + '_opt.mat')
        optftp_path = os.path.join(self.root, osmid + '_optftp.mat')
        
        with h5py.File(sar_path, 'r') as f1:
            x1 = np.float32(f1['x1'])
            x1[x1 < -100] = x1[x1 > -100].min()  # fill missing values with min of non-missing values
            x1 = cv2.resize(x1, dsize=(224, 224), interpolation=cv2.INTER_LINEAR_EXACT)
            p1, p99 = np.percentile(x1, (1, 99))
            x1 = np.clip(x1, p1, p99)
            x1 = (x1 - p1) / (p99 - p1)
            x1 = np.stack((x1, x1, x1), 0)  # 3,224,224
        with h5py.File(sarftp_path, 'r') as f2:
            x2 = np.float32(f2['x2'])
            x2 = cv2.resize(x2, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
            x2 = np.stack((x2, x2, x2), 0)  # 3,224,224
        with h5py.File(opt_path, 'r') as f3:
            x3 = np.float32(f3['x3'])
            x3 = cv2.resize(np.transpose(x3, (1, 2, 0)), dsize=(224, 224), interpolation=cv2.INTER_LINEAR_EXACT)
            x3 = np.transpose(x3, (2, 0, 1))
            p1, p99 = np.percentile(x3, (1, 99))
            x3 = np.clip(x3, p1, p99)
            x3 = (x3 - p1) / (p99 - p1)
        with h5py.File(optftp_path, 'r') as f4:
            x4 = np.float32(f4['x4'])
            x4 = cv2.resize(x4, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
            x4 = np.stack((x4, x4, x4), 0)  # 3,224,224

        # Extract shadow mask from the optical image (x3)
        shadow_mask = extract_shadow_mask(x3)  # shape (1, 224, 224)
        # Option: Combine optical image with shadow mask to form a 4-channel input.
        # For example, you can stack them along the channel dimension:
        opt_with_shadow = np.concatenate((x3, shadow_mask), axis=0)  # now shape (4, 224, 224)

        # You can choose to return the optical image and its shadow mask separately, or together.
        # Here, we return a dictionary including the shadow mask and the 4-channel optical image.
        images = {
            'sar': torch.from_numpy(x1).float(),
            'sarftp': torch.from_numpy(x2).float(),
            'opt': torch.from_numpy(x3).float(),
            'opt_with_shadow': torch.from_numpy(opt_with_shadow).float(),
            'optftp': torch.from_numpy(x4).float()
        }
        # # Optional: sanity check shape
        # if opt_with_shadow.shape[0] != 4:
        #     print(f"[DEBUG] Warning: opt_with_shadow shape is {opt_with_shadow.shape} for ID {osmid}")
        # else:
        #     print(f"[DEBUG] opt_with_shadow shape: {opt_with_shadow.shape}")

        return images, torch.tensor(label).float()
        
    def __len__(self):
        return self.length
    

class MyAug(torch.nn.Module):
    '''
    A custom augmentation module using Kornia.
    '''
    def __init__(self):
        super(MyAug, self).__init__()
        self.k1 = kornia.augmentation.RandomResizedCrop((224, 224), scale=(0.8, 1.0))
        self.k2 = kornia.augmentation.RandomHorizontalFlip(p=0.5)
        self.k3 = kornia.augmentation.RandomVerticalFlip(p=0.5)
        # self.k4 = kornia.augmentation.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5)

    def forward(self, sar, sarftp, opt, optftp):
        sar = self.k1(sar)
        sarftp = self.k1(sar, self.k1._params)
        opt = self.k1(opt, self.k1._params)
        optftp = self.k1(optftp, self.k1._params)

        sar = self.k2(sar)
        sarftp = self.k2(sar, self.k2._params)
        opt = self.k2(opt, self.k2._params)
        optftp = self.k2(optftp, self.k2._params)

        sar = self.k3(sar)
        sarftp = self.k3(sar, self.k3._params)
        opt = self.k3(opt, self.k3._params)
        optftp = self.k3(optftp, self.k3._params)
        
        # sar = self.k4(sar)
        # opt = self.k4(opt)

        return sar, sarftp, opt, optftp
