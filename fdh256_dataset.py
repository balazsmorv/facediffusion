import pathlib
import torch
import numpy as np
from torch.utils.data import Dataset
try:
    import pyspng
    PYSPNG_IMPORTED = True
except ImportError:
    PYSPNG_IMPORTED = False
    print("Could not load pyspng. Defaulting to pillow image backend.")
    from PIL import Image


class FDF256Dataset(Dataset):

    def __init__(self,
                 dirpath,
                 load_keypoints: bool,
                 transform):
        dirpath = pathlib.Path(dirpath)
        self.dirpath = dirpath
        self.transform = transform
        self.load_keypoints = load_keypoints
        assert self.dirpath.is_dir(),\
            f"Did not find dataset at: {dirpath}"
        image_dir = self.dirpath.joinpath("images")
        self.image_paths = list(image_dir.glob("*.png"))
        assert len(self.image_paths) > 0,\
            f"Did not find images in: {image_dir}"
        self.image_paths.sort(key=lambda x: int(x.stem))
        self.landmarks = np.load(self.dirpath.joinpath("landmarks.npy")).reshape(-1, 7, 2).astype(np.float32)
        self.bounding_boxes = torch.from_numpy(np.load(self.dirpath.joinpath("bounding_box.npy")))
        assert len(self.image_paths) == len(self.bounding_boxes)
        assert len(self.image_paths) == len(self.landmarks)
        print(
            f"Dataset loaded from: {dirpath}. Number of samples:{len(self)}")

    def get_mask(self, idx):
        mask = torch.ones((1, 256, 256), dtype=torch.bool)
        bounding_box = self.bounding_boxes[idx]
        x0, y0, x1, y1 = bounding_box
        mask[:, y0:y1, x0:x1] = 0
        return mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        impath = self.image_paths[index]
        if PYSPNG_IMPORTED:
            with open(impath, "rb") as fp:
                im = pyspng.load(fp.read())
        else:
            with Image.open(impath) as fp:
                im = np.array(fp)
        if self.transform is not None:
            im = self.transform(im)
        masks = self.get_mask(index)
        landmark = self.landmarks[index]
        batch = {
            "img": im,
            "mask": masks,
        }
        if self.load_keypoints:
            batch["keypoints"] = landmark
        return batch
