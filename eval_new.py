import glob
import re
import os
import pickle

import nibabel as nib
import numpy as np

from skimage.transform import resize
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure

from midl.ds.AbdomenROIDataset import AbdomenROIDataset


def DSC(gt, pred):
    """
    Dice Similarity Score

    Compute dice similarity score
    :param gt:
    :param pred:
    :return:
    """

    dsc = 2 * np.sum(pred[gt == True]) / (np.sum(gt) + np.sum(pred))
    return dsc


def HD(result, reference, connectivity=1):
    """
    Hausdorff Distance

    Compute
    :return:
    """
    hd1 = surface_distance(result, reference).max()
    hd2 = surface_distance(reference, result).max()
    hd = max(hd1, hd2)

    return hd


def HD95(result, reference, connectivity=1):
    hd1 = surface_distance(result, reference)
    hd2 = surface_distance(reference, result)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)

    return hd95


def ASSD(result, reference, connectivity=1):
    """
    Average Symmetric Surface Distance
    :return:
    """
    assd = np.mean( (ASD(result, reference, connectivity), ASD(reference, result, connectivity)) )

    return assd


def ASD(result, reference, connectivity=1):
    """
    Average Surface Distance
    :param result:
    :param reference:
    :param connectivity:
    :return:
    """
    sds = surface_distance(result, reference, connectivity)
    asd = sds.mean()

    return asd


def sensitivity(result, reference):
    result = result.astype(np.bool)
    reference = reference.astype(np.bool)

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        r = tp / float(tp + fn)
    except ZeroDivisionError:
        r = 0.0

    return r


def precision(result, reference):
    result = result.astype(np.bool)
    reference = reference.astype(np.bool)

    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        p = tp / float(tp + fp)
    except ZeroDivisionError:
        p = 0.0

    return p


def surface_distance(result, reference, connectivity=1):
    result = result.astype(np.bool)
    reference = reference.astype(np.bool)

    # Create binary structure
    struct = generate_binary_structure(result.ndim, connectivity)

    # Test for empty images
    if np.count_nonzero(result) == 0:
        raise RuntimeError('The result image does not contain any binary object.')
    if np.count_nonzero(reference) == 0:
        raise RuntimeError('The reference image does not contain any binary object.')

    # Extract border images
    result_border = result ^ binary_erosion(result, structure=struct)
    reference_border = reference ^ binary_erosion(reference, structure=struct)

    # Compute average surface distance
    dt = distance_transform_edt(~reference_border)
    sds = dt[result_border]

    return sds


def eval(path_gt_ds,
         path_pred_dir,
         metric):
    """
    Evaluate the results of model with the metric.
    :param path_gt_dir:
    :param path_pred_dir:
    :param metric: A function which calculates similarity between ground truth and prediction.
    :return: Mean value of similarity
    """

    with open(path_gt_ds, "rb") as f:
        gt_ds = pickle.load(f)

    val = 0
    for data in gt_ds:
        print(data['name'])
        gt = data['label']
        # gt_idx = int(re.sub('\D', '', os.path.basename(data['name'])))

        # Get a predicted image
        pred = nib.load(os.path.join(path_pred_dir, data['name'])).get_data()

        # Reshape to (D, H, W)
        pred = pred.transpose((-1, 0, 1))

        similarity = metric(gt, pred)
        val += similarity
        print(similarity)

    return val / len(gt_ds)

    # pred_masks = glob.glob(path_pred_dir + '/*')
    #
    # pred_masks.sort(key=lambda f: int(re.sub('\D', '', f)))
    #
    #
    # val = 0
    # for gt, pred in zip(gt_masks, pred_masks):
    #     gt_file = gt
    #     print(gt)
    #     print(pred)
    #     gt = nib.load(gt).get_data()
    #     pred = nib.load(pred).get_data()
    #
    #     # Reshape to (D, H, W)
    #     gt = gt.transpose((-1, 0, 1))
    #     pred = pred.transpose((-1, 0, 1))
    #
    #     gt = resize(gt.astype(int), (64, 128, 128), anti_aliasing=False, order=0, preserve_range=True)
    #     gt = (gt == 6)
    #
    #     gt_idx = int(re.sub('\D', '', os.path.basename(gt_file)))
    #     if gt_idx < 44:
    #         gt = gt[::-1]
    #
    #     similarity = metric(gt, pred)
    #     val += similarity
    #     print(similarity)
    #
    # return val / len(gt_masks)


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
    path_gt_dir = 'D:/2020_INFINITT/INFINITT_Deep/utils/test_roi_ds'
    path_pred_dir = 'E:/Data/INFINITT/Results/TestNet'

    eval_dsc    = eval(path_gt_ds=path_gt_dir,
                       path_pred_dir=path_pred_dir,
                       metric=DSC)
    eval_hd     = eval(path_gt_ds=path_gt_dir,
                       path_pred_dir=path_pred_dir,
                       metric=HD95)
    eval_assd   = eval(path_gt_ds=path_gt_dir,
                       path_pred_dir=path_pred_dir,
                       metric=ASSD)
    eval_s      = eval(path_gt_ds=path_gt_dir,
                       path_pred_dir=path_pred_dir,
                       metric=sensitivity)
    eval_p      = eval(path_gt_ds=path_gt_dir,
                       path_pred_dir=path_pred_dir,
                       metric=precision)
    print("DSC:", eval_dsc)
    print("HD:", eval_hd)
    print("ASSD:", eval_assd)
    print("Sensitivity:", eval_s)
    print("Precision:", eval_p)

