import pickle
import os

import nibabel as nib
import numpy as np


def save_cropping_images(ds, dst):
    for data in ds:
        filename = data['name']
        filepath = os.path.join(dst, filename)

        print(data['roi'])
        res = nib.Nifti1Image(data['image'].transpose((1, 2, 0)), np.eye(4))
        nib.save(res, filepath)

# TODO: validate the below func.
def vis_compare(ds, path_pred):
    dst = os.path.join(path_pred, "cmp")
    os.mkdir(dst)

    for data in ds:
        filename = data['name']
        filepath = os.path.join(dst, filename)

        gt = data['label']
        pred = nib.load(os.path.join(path_pred, filename)).get_data()

        # Reshape to (D, H, W)
        pred = pred.transpose((-1, 0, 1))

        assert(pred.shape == gt.shape)

        pred *= 5
        gt *= 10

        out = pred + gt
        out = nib.Nifti1Image(out.transpose((1, 2, 0)), np.eye(4))
        nib.save(out, filepath)


if __name__=="__main__":

    with open("./utils/test_roi_ds", "rb") as f:
        gt_ds = pickle.load(f)

    dst = "E:/Data/INFINITT/Integrated/cropped"
    save_cropping_images(gt_ds, dst)