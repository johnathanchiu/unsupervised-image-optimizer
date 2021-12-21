#  IMAGE LOADER

from .helper import *

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os, cv2, random
from PIL import Image 


def sample_img(img, n_samples=32):
    assert n_samples % 4 == 0
    partitions = partition(img)
    partitions.sort(reverse=True, key=lambda x: np.var(x))
    partitions_large = partitions[:len(partitions)//2]
    partitions_small = partitions[len(partitions)//2:]
    # random sample then swap channels and samples axes
    samples_a = random.sample(partitions_large, 3 * (n_samples // 4))
    samples_b = random.sample(partitions_small, n_samples // 4)
    samples = samples_a + samples_b
    random.shuffle(samples)
    samples = torch.tensor(np.array(samples), dtype=torch.double)
    return samples

def read_img(file_path, convert=False):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    # channels first
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img, -1, 0)
    else:
        img = np.expand_dims(img, axis=0)
    img = img.astype(np.float64)
    if convert:
        img = rgb_ycbcr(img)
    img -= 128
    return img

def normalize(tensor):
    return (tensor - torch.mean(tensor)) / torch.std(tensor)

class ImageCompressionDataset(Dataset):
    def __init__(self, img_path, device, crop=(344, 344), samples=64):
        self.samples = samples
        self.img_path = img_path
        self.crop = crop
        self.device = device

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = read_img(self.img_path, False)
        c, x, y = img.shape
        assert x >= self.crop[0] and y >= self.crop[1]
        assert x * y / (8 * 8) >= self.samples
        spatial_samples = sample_img(img, self.samples)
        freq_samples = dct2(spatial_samples)
        cx, cy = self.crop
        x1, y1 = random.randint(0, x - cx - 1), random.randint(0, y - cy - 1)
        x2, y2 = x1 + cx, y1 + cy
        img = img[:,x1:x2,y1:y2]
        partition_freq = dct2(torch.tensor(partition_inplace(img), 
                                           dtype=torch.double, 
                                           device=self.device))
        return freq_samples, partition_freq, img + 128, self.img_path

def save_test_image_color(qtable, path_load, folder_save, n):
    file_save =  os.path.join(folder_save, f'{n}.jpg')
    test_tables = np.round(zz_encode(qtable).detach().cpu().numpy()).astype(int)[0]
    table1, table2 = test_tables
    im1 = plt.imread(path_load)
    img = Image.fromarray(im1)
    img.save(file_save, qtables={0: table1, 1: table2}, optimize=False)

def save_test_image(qtable, path_load, folder_save, n):
    file_save =  os.path.join(folder_save, f'{n}.jpg')
    test_table = np.round(torch.squeeze(zz_encode(qtable)).detach().cpu().numpy()).astype(int)
    im1 = plt.imread(path_load)
    img = Image.fromarray(im1)
    img.save(file_save, qtables={0: test_table}, optimize=False)