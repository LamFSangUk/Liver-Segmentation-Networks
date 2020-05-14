import os
import glob
import nibabel as nib
from torch.utils.data import Dataset
from skimage.transform import resize

import matplotlib.pyplot as plt
import numpy as np


class AbdomenDataset(Dataset):
    def __init__(self,
                 organ,
                 width,
                 height,
                 depth,
                 path_image_dir,
                 path_label_dir):
        """
        AbdomenDataset is a dataset including abdomen images.
        :param path_image: Directory path for images
        :param path_label: Directory path for labels
        """
        self.shape = (depth, height, width)
        self.paths_image = glob.glob(path_image_dir + '/*')
        self.paths_label = glob.glob(path_label_dir + '/*')

        self.label_num = {
            "spleen" : 1,
            "right kidney" : 2,
            "left kidney" : 3,
            "gallbladder" : 4,
            "esophagus" : 5,
            "liver" : 6,
            "stomach" : 7,
            "aorta" : 8,
            "inferior vena cava" : 9,
            "portal vein and splenic vein" : 10,
            "pancreas" : 11,
            "right adrenal gland" : 12,
            "left adrenal gland" : 13,
            "duodenum" : 14
        }

        self.organ = organ

        assert(len(self.paths_image) == len(self.paths_label))
        self.n_data = len(self.paths_image)

        # Load images and resize them.
        self.image_stack = np.empty((0, depth, height, width))
        self.label_stack = np.empty((0, depth, height, width))

        # Save filename list
        self.filename = []

        for i in range(len(self.paths_image)):
            self.filename.append(os.path.basename(self.paths_image[i]))
            # shape will be (H, W, D)
            nib_image_file = nib.load(self.paths_image[i])
            image = nib_image_file.get_data()
            label = nib.load(self.paths_label[i]).get_data()

            # Reshape to (D, H, W)
            image = image.transpose((-1, 0, 1))
            label = label.transpose((-1, 0, 1))

            image = resize(image.astype(float), self.shape)
            label = resize(label.astype(int), self.shape, anti_aliasing=False, order=0, preserve_range=True)

            image = image.astype(np.float32)
            label = label.astype(np.uint8)

            # choose an organ
            label = (label == self.label_num[self.organ])

            # Save to stacks
            self.image_stack = np.vstack((self.image_stack, np.expand_dims(image, axis=0)))
            self.label_stack = np.vstack((self.label_stack, np.expand_dims(label, axis=0)))

        self.label_mean = np.mean(self.label_stack)

        print("Data Loaded")
        print()

    def __len__(self):
        return self.n_data

    def __getitem__(self, item):
        # # shape will be (H, W, D)
        # nib_image_file = nib.load(self.paths_image[item])
        # image = nib_image_file.get_data()
        # label = nib.load(self.paths_label[item]).get_data()
        #
        # # Reshape to (D, H, W)
        # image = image.transpose((-1, 0, 1))
        # label = label.transpose((-1, 0, 1))
        #
        # image = resize(image.astype(float), self.shape)
        # label = resize(label.astype(int), self.shape, anti_aliasing=False, order=0, preserve_range=True)
        #
        # image = image.astype(np.float32)
        # label = label.astype(np.uint8)
        #
        # # choose an organ
        # label = (label == self.label_num[self.organ])
        #

        sample = {'image': self.image_stack[item],
                  'label': self.label_stack[item],
                  'name': self.filename[item]}

        return sample

if __name__=="__main__":
    ds = AbdomenDataset("liver", 128,128,64,
                   path_image_dir="E:/Data/INFINITT/Integrated/train/img",
                   path_label_dir="E:/Data/INFINITT/Integrated/train/label")

    for data in ds:
        pass