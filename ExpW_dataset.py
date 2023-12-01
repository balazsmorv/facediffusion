from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
import pandas as pd
import torchvision.transforms.functional as TF


class FacialExpressionsDataset(Dataset):

    def __init__(self, csv_file: str, root_dir: str, transform=None):
        self.labels = pd.read_csv(csv_file, index_col='idx')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = Image.open(image_path)
        image_tensor = TF.to_tensor(image)
        
        label = self.labels.iloc[idx, 1].astype('int')
        impath = self.labels.iloc[idx, 0]
        
        keypoints = torch.tensor(data=[self.labels.iloc[idx, 3:]])
        keypoints = torch.reshape(keypoints, shape=(7, 2))

        sample = {'image': image_tensor, 'label': label, 'keypoints': keypoints, 'impath': impath, 'idx': idx}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    

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

    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file, index_col='idx')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = Image.open(image_path)
        image_tensor = TF.to_tensor(image)
        
        label = self.labels.iloc[idx, 1].astype('int')
        impath = self.labels.iloc[idx, 0]
        
        
        keypoints = torch.tensor(data=[self.labels.iloc[idx, 3:17]])
        keypoints = torch.reshape(keypoints, shape=(7, 2))
        
        bounding_boxes = torch.tensor(data=self.labels.iloc[idx, 17:])

        sample = {'image': image_tensor, 'label': label, 'keypoints': keypoints, 'impath': impath, 'idx': idx, 'bbox': bounding_boxes}

        if self.transform:
            sample = self.transform(sample)

        return sample