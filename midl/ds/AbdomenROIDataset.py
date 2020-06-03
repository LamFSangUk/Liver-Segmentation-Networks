import re
import os
import glob
import random
import csv

import nibabel as nib
import numpy as np

from torch.utils.data import Dataset
from skimage.transform import resize
from skimage.exposure import equalize_adapthist
import scipy.ndimage

from midl.utils.image_processing import create_gaussian_heatmap
from midl.utils.image_processing import add_gaussian_noise, cutout, generate_random_rotation_matrix



class AbdomenROIDataset(Dataset):
    def __init__(self,
                 organ,
                 width,
                 height,
                 depth,
                 path_image_dir,
                 path_label_dir,
                 path_roi,
                 aug=False):

        self.shape = (depth, height, width)
        self.paths_image = glob.glob(path_image_dir + '/*')
        self.paths_label = glob.glob(path_label_dir + '/*')

        assert(len(self.paths_image) == len(self.paths_label))

        self.label_num = {
            "spleen": 1,
            "right kidney": 2,
            "left kidney": 3,
            "gallbladder": 4,
            "esophagus": 5,
            "liver": 6,
            "stomach": 7,
            "aorta": 8,
            "inferior vena cava": 9,
            "portal vein and splenic vein": 10,
            "pancreas": 11,
            "right adrenal gland": 12,
            "left adrenal gland": 13,
            "duodenum": 14
        }

        self.n_data = len(self.paths_image)

        self.organ = organ

        self.aug = aug

        # Parsing csv
        roi_dict = {}
        skip_header = False
        with open(path_roi, 'r', encoding='utf-8') as f:
            rdr = csv.reader(f)
            for line in rdr:

                # skip the first line
                if skip_header is False:
                    skip_header = True
                    continue

                # HWD(y,x,z) => DHW(z,y,x)
                min_point = (int(line[8]), int(line[4]), int(line[6]))
                max_point = (int(line[9]), int(line[5]), int(line[7]))
                roi = {"min_p": min_point, "max_p": max_point}
                roi_dict[int(line[0])] = roi

        # Create image and roi stack.
        self.image_stack = np.empty((0, depth, height, width))
        self.label_stack = np.empty((0, depth, height, width))

        self.roi_stack = []
        self.roi_heatmap_stack = np.empty((0, 2, depth, height, width))

        # Create file name list
        self.filename = []

        # Preprocess images
        for i in range(len(self.paths_image)):
            # Save filename
            self.filename.append(os.path.basename(self.paths_image[i]))

            # Load a image and a label
            # shape will be (H, W, D)
            nib_image_file = nib.load(self.paths_image[i])
            image = nib_image_file.get_data()
            label = nib.load(self.paths_label[i]).get_data()


            # Reshape to (D, H, W)
            image = image.transpose((-1, 0, 1))
            image_shape = image.shape
            image_idx = int(re.sub('\D', '', os.path.basename(self.paths_image[i])))

            label = label.transpose((-1, 0, 1))

            if image_idx < 44:
                label = label[::-1]

            # Crop
            roi = roi_dict[image_idx]
            min_p = roi['min_p']
            max_p = roi['max_p']
            image = image[min_p[0]:max_p[0], min_p[1]:max_p[1], min_p[2]:max_p[2]]
            label = label[min_p[0]:max_p[0], min_p[1]:max_p[1], min_p[2]:max_p[2]]

            # Resize
            image = resize(image.astype(float), self.shape)
            label = resize(label.astype(int), self.shape, anti_aliasing=False, order=0, preserve_range=True)
            roi['min_p'] = np.array(roi['min_p']) * np.array((depth / image_shape[0],
                                                              height / image_shape[1],
                                                              width / image_shape[2]))

            roi['max_p'] = np.array(roi['max_p']) * np.array((depth / image_shape[0],
                                                              height / image_shape[1],
                                                              width / image_shape[2]))

            heatmap = np.empty((2, depth, height, width))
            heatmap[0] = create_gaussian_heatmap(roi['min_p'][::-1], self.shape[::-1]).transpose((-1, 1, 0))
            heatmap[1] = create_gaussian_heatmap(roi['max_p'][::-1], self.shape[::-1]).transpose((-1, 1, 0))
            self.roi_stack.append(roi)

            # Clip and Normalize
            min_val = np.min(image)
            if min_val > -1020:
                image = np.clip(image, -1000, 3095)
                image -= 24
            else:
                image = np.clip(image, -1024, 3071)

            image = np.clip(image, -340, 360)
            image = (image + 340) / (340 + 360)

            # CLAHE
            image = equalize_adapthist(image)

            image = image.astype(np.float32)
            label = label.astype(np.uint8)

            # choose an organ
            label = (label == self.label_num[self.organ])

            # Save to stacks
            self.image_stack = np.vstack((self.image_stack, np.expand_dims(image, axis=0)))
            self.label_stack = np.vstack((self.label_stack, np.expand_dims(label, axis=0)))
            self.roi_heatmap_stack = np.vstack((self.roi_heatmap_stack, np.expand_dims(heatmap, axis=0)))

        # Save a mean value of labels.
        self.label_mean = np.mean(self.label_stack)

        print("Data Loaded")
        print()

    def __len__(self):
        return self.n_data

    def __getitem__(self, item):

        image = self.image_stack[item]
        label = self.label_stack[item]
        roi = self.roi_stack[item]
        roi_heatmap = self.roi_heatmap_stack[item]

        if self.aug:
            if random.SystemRandom().random() > 0.3:
                image = add_gaussian_noise(image)
                image = (image - np.min(image)) / (np.max(image) - np.min(image))

            if random.SystemRandom().random() > 0.5:
                image = cutout(image)

        sample = {'image':image,
                  'label':label,
                  'roi': roi,
                  'roi_heatmap': roi_heatmap,
                  'name': self.filename[item]}

        return sample


if __name__ == "__main__":
    ds = AbdomenROIDataset(128, 128, 64,
                           path_image_dir="E:/Data/INFINITT/Integrated/debug/img",
                           path_roi="D:/2020_INFINITT/cropping.csv")

    for data in ds:
        heatmaps = data['roi_heatmap']

        # For debug
        res = nib.Nifti1Image(heatmaps[0].transpose((1, 2, 0)), np.eye(4))
        nib.save(res, '%s' % "test")