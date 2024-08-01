import torch
from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import random as rd
import cv2
import config

feature_size_h = config.feature_size_h
feature_size_w = config.feature_size_w
stride_h = config.stride_h
stride_w = config.stride_w


def get_mean_std(loader):
    # VAR[X]=E[X**2]-E(X)**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, _0 in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def make_dataset(root):
    L_img = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                L_img.append(os.path.join(root, os.path.splitext(file)[0] + '.jpg'))
    return L_img


def make_dataset_d(root):
    L_img = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                L_img.append(os.path.join(root, os.path.splitext(file)[0] + '.png'))
    return L_img


def make_dataset_txt(root):
    L_label = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L_label.append(os.path.join(root, os.path.splitext(file)[0] + '.txt'))
    return L_label


def mask_to_onehot(mask):
    # Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    # hot encoding vector, C is usually 1 or 3, and K is the number of class
    palette = [[0], [128], [255]]
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


class LiverDataset3(Dataset):
    def __init__(self, root, transform=None, is_train=True):
        imgs = make_dataset(root + '/images')
        depthes = make_dataset_d(root + '/depthes')
        masks = make_dataset_d(root + '/segmentation')
        labels = make_dataset_txt(root + '/final_labels32asymmetric')
        labels_flip = make_dataset_txt(root + '/final_labels_flip32asymmetric')
        self.imgs = imgs
        self.depthes = depthes
        self.masks = masks
        self.labels = labels
        self.labels_flip = labels_flip
        self.transform = transform
        self.is_train = is_train

    def __getitem__(self, index):
        # flip or not
        flip = rd.randint(0, 1)
        x_path = self.imgs[index]
        depth_path = self.depthes[index]
        label_path = self.labels[index]
        mask_path = self.masks[index]
        if self.is_train:
            if flip == 1:
                label_path = self.labels_flip[index]

        # read txt files
        loc_b = np.zeros((feature_size_h, feature_size_w, 4))  # labels of location
        cls_b = np.zeros((feature_size_h, feature_size_w, 1))  # labels of probability of Gaussian
        heatmap_b = np.zeros((feature_size_h, feature_size_w, 1))  # labels of classification
        loc_r = np.zeros((feature_size_h, feature_size_w, 4))
        cls_r = np.zeros((feature_size_h, feature_size_w, 1))
        heatmap_r = np.zeros((feature_size_h, feature_size_w, 1))
        with open(label_path) as f:
            bbox = f.read().split('\n')
        bbox = [x.split() for x in bbox]
        bbox = [float(x) for y in bbox for x in y]
        for i in range(len(bbox) // 8):
            i_loc = int(bbox[i * 8 + 6])
            j_loc = int(bbox[i * 8 + 7])
            if int(bbox[i * 8]) == 0:
                cls_b[i_loc, j_loc] = bbox[i * 8 + 5]
                heatmap_b[i_loc, j_loc] = 1
                for j in range(4):
                    loc_b[i_loc, j_loc, j] = bbox[i * 8 + 1 + j]
            if int(bbox[i * 8]) == 1:
                cls_r[i_loc, j_loc] = bbox[i * 8 + 5]
                heatmap_r[i_loc, j_loc] = 1
                for j in range(4):
                    loc_r[i_loc, j_loc, j] = bbox[i * 8 + 1 + j]
        # read images
        img_x = Image.open(x_path)  # RGB image
        img_d = Image.open(depth_path)  # Depth image
        img_m = cv2.imread(mask_path)  # mask image
        if self.is_train:
            # random flip
            if flip == 1:
                img_x = img_x.transpose(Image.FLIP_LEFT_RIGHT)
                img_d = img_d.transpose(Image.FLIP_LEFT_RIGHT)
                img_m = cv2.flip(img_m, 1)

        img_m = mask_to_onehot(img_m)
        # To tensor
        loc_b = transforms.ToTensor()(loc_b)
        cls_b = transforms.ToTensor()(cls_b)
        heatmap_b = transforms.ToTensor()(heatmap_b)
        loc_r = transforms.ToTensor()(loc_r)
        cls_r = transforms.ToTensor()(cls_r)
        heatmap_r = transforms.ToTensor()(heatmap_r)
        img_m = transforms.ToTensor()(img_m)
        img_d = transforms.ToTensor()(img_d)
        if self.transform is not None:
            img_x = self.transform(img_x)

        return img_x, img_d, (cls_b, loc_b, heatmap_b), (cls_r, loc_r, heatmap_r), img_m

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    liver_dataset = LiverDataset3("data/train", transform=transforms.ToTensor(), is_train=False)
    dataloaders = DataLoader(liver_dataset, batch_size=32, shuffle=True, num_workers=4)

    mean, std = get_mean_std(dataloaders)
    print(mean)
    print(std)
