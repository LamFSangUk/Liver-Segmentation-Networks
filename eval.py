import numpy as np
from skimage.transform import resize
import nibabel as nib
import glob
import re

def DSC(gt, pred):
    """
    Calculate dice similarity score
    :param gt:
    :param pred:
    :return:
    """
    dsc = 2 * np.sum(pred[gt == True]) / (np.sum(gt) + np.sum(pred))
    return dsc


def HD():
    pass


def eval(path_gt_dir,
         path_pred_dir,
         metric):
    """
    Evaluate the results of model with the metric.
    :param path_gt_dir:
    :param path_pred_dir:
    :param metric: A function which calculates similarity between ground truth and prediction.
    :return: Mean value of similarity
    """

    gt_masks = glob.glob(path_gt_dir + '/*')
    pred_masks = glob.glob(path_pred_dir + '/*')

    gt_masks.sort(key=lambda f: int(re.sub('\D', '', f)))
    pred_masks.sort(key=lambda f: int(re.sub('\D', '', f)))

    assert(len(gt_masks) == len(pred_masks))

    val = 0
    for gt, pred in zip(gt_masks, pred_masks):
        print(gt)
        print(pred)
        gt = nib.load(gt).get_data()
        pred = nib.load(pred).get_data()

        gt = resize(gt.astype(int), (128, 128, 64), anti_aliasing=False, order=0, preserve_range=True)
        gt = (gt == 6)

        similarity = metric(gt, pred)
        val += similarity
        print(similarity)

    return val / len(gt_masks)


if __name__ == "__main__":

    # gt = nib.load('E:/Data/INFINITT/Integrated/test/label/mask_2.nii.gz').get_data()
    #
    # gt = resize(gt.astype(int), (128, 128, 64), anti_aliasing=False, order=0, preserve_range=True)
    # gt = (gt == 6)
    #
    # pred = nib.load('D:/2020_INFINITT/INFINITT_Deep/0.nii.gz').get_data()
    #
    # print(gt.shape)
    #
    # dsc = DSC(gt, pred)
    # print(dsc)

    evaluated = eval(path_gt_dir='E:/Data/INFINITT/Integrated/test/label',
                     path_pred_dir='E:/Data/INFINITT/Results/VoxResNet',
                     metric=DSC)
    print("Final:", evaluated)