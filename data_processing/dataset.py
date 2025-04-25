import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import cv2
import copy
from torch.utils.data import DataLoader


def list_directories(target_dir):
    return [name for name in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, name))]


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class LFADataset(Dataset):
    def __init__(self, main_dir, series, 
                 startpoint=10, endpoint=900, steps=90,
                 resize_shape=48, shuffle=True,
                 split=0.8, mode='train', 
                 add_noise=True, noise_mean=0., noise_std=1.):
        super(LFADataset, self).__init__()
        assert mode in ['train', 'val', 'test']
        if mode == 'test':
            assert split == 0.0

        assert startpoint / 10 <= series <= endpoint / 10
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.steps = steps
        self.series = series
        self.resize_shape = (resize_shape, resize_shape)
        
        self.add_noise = add_noise
        if self.add_noise:
            self.noise = AddGaussianNoise(noise_mean, noise_std)  # Initialize noise module
        
        image_list = np.linspace(startpoint, endpoint, steps, endpoint=True, dtype=int).tolist()
        self.image_list = [f'{image_name}.png' for image_name in image_list]
        self.concentration_dirs = list_directories(main_dir)
        self.concentrations = [float(con) for con in self.concentration_dirs]
        self.max_con = np.max(self.concentrations)
        self.sample_dirs = []
        self.dir_labels = []
        for con_dir, con in zip(self.concentration_dirs, self.concentrations):
            cur_dir = os.path.join(main_dir, con_dir)
            sub_dirs = list_directories(cur_dir)
            for sub_dir in sub_dirs:
                sample_dir = os.path.join(cur_dir, sub_dir)
                self.sample_dirs.append(sample_dir)
                self.dir_labels.append(con)

        self.indices = np.arange(len(self.sample_dirs), dtype=int)
        if shuffle:  #* For splitting training and validation set
            random.shuffle(self.indices)

        if split > 0:
            self.train_indices = self.indices[:int(self.indices.shape[0] * split)]
            self.val_indices = self.indices[int(self.indices.shape[0] * split):]

        self.mode = mode
        self.cur_dir = None
        
    def _mode_to(self, mode):
        assert mode in ['train', 'val', 'test']
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return self.train_indices.shape[0]
        elif self.mode == 'val':
            return self.val_indices.shape[0]
        else:
            return len(self.sample_dirs)

    def __getitem__(self, idx):
        if self.mode == 'train':
            target_idx = self.train_indices[idx]
        elif self.mode == 'val':
            target_idx = self.val_indices[idx]
        else:
            target_idx = self.indices[idx]

        y = self.dir_labels[target_idx]
        y = torch.tensor(y, dtype=torch.float32) / self.max_con

        t = np.linspace(self.startpoint, self.endpoint, self.steps, endpoint=True, dtype=int).tolist()[:self.series]
        image_list = [f'{img_idx}.png' for img_idx in t]
        image_dir = self.sample_dirs[target_idx]
        self.cur_dir = image_dir

        t = torch.tensor(t, dtype=torch.float32)
        x = torch.zeros(self.series, *self.resize_shape, 6)
        for i, image_name in enumerate(image_list):
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.resize_shape)

            image = image.astype(np.uint8)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

            rgb_image = rgb_image / 255.0
            hsv_image = hsv_image / 255.0

            x[i, :, :, :3] = torch.from_numpy(rgb_image).float()
            x[i, :, :, 3:] = torch.from_numpy(hsv_image).float()

        x = x.permute(0, 3, 1, 2)
        if self.add_noise:
            x = self.noise(x)
        return x, t, y


def build_dataset(dataset_dir, series, batch_size,
                  startpoint, endpoint, steps,
                  resize_shape=(48, 48),
                  shuffle=True, split=[0.7, 0.9], noise_mean=0., noise_std=1.):
    # training and validation set
    cur_dir = os.path.join(dataset_dir, 'train')
    main_dir = os.path.join(cur_dir)
    train_set = LFADataset(main_dir=main_dir, series=series,
                           resize_shape=resize_shape,
                           startpoint=startpoint, endpoint=endpoint, steps=steps,
                           shuffle=shuffle, split=split,
                           mode='train', noise_mean=noise_mean, noise_std=noise_std)
    val_set = copy.deepcopy(train_set)
    val_set._mode_to('val')

    # test set
    cur_dir = os.path.join(dataset_dir, 'test')
    main_dir = os.path.join(cur_dir)
    test_set = LFADataset(main_dir=main_dir, series=series,
                          resize_shape=resize_shape,
                          startpoint=startpoint, endpoint=endpoint, steps=steps,
                          shuffle=False, split=0.0, mode='test', 
                          noise_mean=noise_mean, noise_std=noise_std)

    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_set = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)
    test_set = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_set, val_set, test_set