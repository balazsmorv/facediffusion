from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
import pandas as pd
import numpy as np


class FacialExpressionsWithKeypointsDataset(Dataset):
    """
    Dataset class for http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html
    Expression label：
        "0" "angry"
        "1" "disgust"
        "2" "fear"
        "3" "happy"
        "4" "sad"
        "5" "surprise"
        "6" "neutral"
    """

    def __init__(self, csv_file, root_dir, img_transform: torch.nn.Module, mask_transform: torch.nn.Module = None):
        self.labels = pd.read_csv(csv_file, index_col='idx')
        self.root_dir = root_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.labels)

    def get_mask(self, idx):
        mask = torch.ones((1, 96, 96), dtype=torch.bool)
        bounding_box = self.labels.iloc[idx, 17:] * 96
        x0, y0, x1, y1 = bounding_box
        mask[:, int(y0):int(y1), int(x0):int(x1)] = 0
        return mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        with Image.open(image_path) as fp:
            im = np.array(fp)
        image = self.img_transform(im)
        
        label = self.labels.iloc[idx, 1].astype('int')
        one_hot_label = torch.zeros(size=(7,))
        one_hot_label[label-1] = 1
        impath = self.labels.iloc[idx, 0]

        masks = self.get_mask(idx)
        if self.mask_transform is not None:
            masks = self.mask_transform(masks)
        
        keypoints = torch.tensor(data=[self.labels.iloc[idx, 3:13]], dtype=torch.float)
        keypoints = torch.squeeze(keypoints)

        bounding_boxes = torch.tensor(data=self.labels.iloc[idx, 17:])

        sample = {'image': image,
                  'label': one_hot_label,
                  'keypoints': keypoints,
                  'impath': impath,
                  'idx': idx,
                  'bbox': bounding_boxes,
                  'mask': masks}

        return sample