from __future__ import print_function

import glob

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class RecipeLoader(Dataset):
    def __init__(self, filename: str, is_train: bool = True, measurement=None):
        # Get the data file names
        if is_train:
            self.datafiles = glob.glob(filename + "/[0-9a-b]*.jpg")
        else:
            self.datafiles = glob.glob(filename + "/[c-z]*.jpg")

        self.total = len(self.datafiles)
        self.output_height = 64
        self.output_width = 64
        self.measurement = measurement
        self.transforms = transforms.Compose([
            transforms.Resize([self.output_height, self.output_width], 2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):

        batch_file = self.datafiles[index]
        try:
            x_real = pil_loader(batch_file)
        except:
            return None
        x_real = self.transforms(x_real)
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
