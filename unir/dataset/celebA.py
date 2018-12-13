import glob

import imageio
import numpy as np
import torch
from skimage.transform import resize, rotate
from torch.utils.data.dataset import Dataset


def imread(path):
    return imageio.imread(path).astype(np.float)


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return resize(x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w], mode='constant', anti_aliasing=True)


def transform(image, input_height, input_width, resize_height=64, resize_width=64, angle=90, is_crop=True):
    if is_crop:
        cropped_image = center_crop(image, input_height, input_width,
                                    resize_height, resize_width)
    else:
        cropped_image = resize(
            image, [resize_height, resize_width], mode='constant', anti_aliasing=True)
    cropped_image = rotate(cropped_image, angle)
    return np.array(cropped_image) / 127.5 - 1.


def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, angle=90, is_crop=True):
    image = imread(image_path)
    return transform(image, input_height, input_width, resize_height, resize_width, angle, is_crop)


class CelebALoader(Dataset):
    def __init__(self, filename: str, is_train: bool = True, measurement=None):
        # Get the data file names

        if is_train:
            self.datafiles = glob.glob(filename + '/?[0-7]*.jpg')
        else:
            self.datafiles = glob.glob(filename + '/?[8-9]*.jpg')

        self.total = len(self.datafiles)
        # Set the pointer to initial location
        # Options for reading the files
        self.input_height = 108
        self.input_width = 108
        self.output_height = 64
        self.output_width = 64
        self.is_crop = True
        self.measurement = measurement
        if not rotate:
            self.angle = 90
        else:
            self.angle = 0

    def __getitem__(self, index):

        batch_file = self.datafiles[index]
        x_real = get_image(batch_file,
                           input_height=self.input_height,
                           input_width=self.input_width,
                           resize_height=self.output_height,
                           resize_width=self.output_width,
                           is_crop=self.is_crop, angle=self.angle)

        x_real = torch.tensor(x_real, dtype=torch.float).permute(2, 0, 1)
        x_measurement = x_real.unsqueeze(0)

        meas = self.measurement.measure(x_measurement, device='cpu', seed=index)
        dict_var = {
            'sample': x_real,
        }
        dict_var.update(meas)
        if 'mask' in dict_var:
            dict_var['mask'] = dict_var['mask'][0]
        dict_var["measured_sample"] = dict_var["measured_sample"][0]  # because the dataloader add a dimension
        return dict_var

    def __len__(self):
        return self.total
